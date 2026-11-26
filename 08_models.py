"""
Model builders for the four-backbone benchmark (3-class CN/MCI/AD).

  convnext : ConvNeXt-Tiny from timm (ImageNet-pretrained)
  mambad   : bidirectional Vision-Mamba, reference selective scan
  vil      : Vision-xLSTM from NX-AI/vision-lstm
  vrwkv    : Vision-RWKV, plain-PyTorch reference implementation (OpenGVLab/Vision-RWKV)

MambAD and Vision-RWKV use plain-PyTorch operators (no custom CUDA) so they run on a
single T4 GPU. Each builder returns an nn.Module mapping (B,3,224,224) -> (B,num_classes)
and exposes get_classifier() for the head warm-up helper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# ConvNeXt  (timm, pretrained)                                    carry-forward
# ============================================================================
def build_convnext(num_classes=3, pretrained=True, **kw):
    import timm
    m = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes)
    return m


# ============================================================================
# MambAD  (bidirectional Vision-Mamba, reference scan)            carry-forward
#   h_t = exp(dt_t·A)·h_{t-1} + (dt_t·B_t)·x_t ;  y_t = C_t·h_t + D·x_t
# ============================================================================
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.d_inner = expand * d_model
        self.dt_rank = max(1, math.ceil(d_model / 16))
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # dt initialisation (critical for stability - see reference Mamba)
        dt_min, dt_max, dt_init_floor = 1e-3, 1e-1, 1e-4
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        self.dt_proj.weight._no_reinit = True
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)); self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner)); self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _scan(self, x, dt, B, C):
        b, l, di = x.shape
        A = -torch.exp(self.A_log.float())
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dBx = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
        h = torch.zeros(b, di, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(l):
            h = dA[:, t] * h + dBx[:, t]
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
        return torch.stack(ys, dim=1) + x * self.D

    def forward(self, x):
        b, l, _ = x.shape
        xs, z = self.in_proj(x).chunk(2, dim=-1)
        xs = self.conv1d(xs.transpose(1, 2))[:, :, :l].transpose(1, 2)
        xs = F.silu(xs)
        with torch.amp.autocast("cuda", enabled=False):
            xs32 = xs.float()
            dt, Bm, Cm = torch.split(self.x_proj(xs32),
                                     [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = F.softplus(self.dt_proj(dt))
            y = self._scan(xs32, dt, Bm, Cm)
        return self.out_proj(y.to(z.dtype) * F.silu(z))


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = SelectiveSSM(d_model, d_state=d_state, expand=expand)
        self.bwd = SelectiveSSM(d_model, d_state=d_state, expand=expand)

    def forward(self, x):
        h = self.norm(x)
        return x + self.fwd(h) + self.bwd(h.flip(1)).flip(1)


class MambAD(nn.Module):
    def __init__(self, num_classes=3, img_size=224, patch_size=32,
                 dim=192, depth=4, d_state=16, expand=2, in_chans=3):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, dim, patch_size, patch_size)
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [BiMambaBlock(dim, d_state=d_state, expand=expand) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            if not getattr(m.weight, "_no_reinit", False):
                nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None and not getattr(m.bias, "_no_reinit", False):
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def get_classifier(self):
        return self.head

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x).mean(dim=1))


def build_mambad(num_classes=3, patch_size=32, dim=192, depth=4, d_state=16, **kw):
    return MambAD(num_classes=num_classes, patch_size=patch_size, dim=dim,
                  depth=depth, d_state=d_state)


# ============================================================================
# Vision-xLSTM (ViL)  - official NX-AI implementation             NOVEL for AD
#   pip install vision_lstm   (github.com/NX-AI/vision-lstm)
# ============================================================================
def build_vil(num_classes=3, img_size=224, patch_size=16, dim=192, depth=12, **kw):
    try:                                      # package layout (folder copied in / installed)
        from vision_lstm import VisionLSTM2
    except ImportError:
        try:                                  # module layout (vision_lstm/vision_lstm.py)
            from vision_lstm.vision_lstm import VisionLSTM2
        except ImportError as e:
            raise ImportError(
                "Vision-xLSTM (ViL) not found. It is not on PyPI - install the official "
                "code from GitHub, e.g.:\n"
                "  !git clone https://github.com/NX-AI/vision-lstm.git /kaggle/tmp/vil\n"
                "  !cp -r /kaggle/tmp/vil/vision_lstm /kaggle/working/\n"
                "  !pip install kappamodules einops\n"
                "(github.com/NX-AI/vision-lstm)") from e
    return VisionLSTM2(
        dim=dim, depth=depth, patch_size=patch_size,
        input_shape=(3, img_size, img_size), output_shape=(num_classes,),
        pooling="bilateral_flatten", drop_path_rate=0.0,
    )


# ============================================================================
# Vision-RWKV  - self-contained reference implementation          NOVEL for AD
#   architecture per OpenGVLab/Vision-RWKV (ICLR 2025): patch embed ->
#   N x [ spatial-mix (2D Q-shift + bidirectional WKV) , channel-mix ] -> head.
#   The WKV operator is the numerically-stable RWKV recurrence in plain PyTorch
#   (run forwards and backwards and averaged, since image tokens are non-causal).
# ============================================================================
def q_shift_2d(x, H, W):
    """Vision-RWKV Q-shift: shift a quarter of the channels each of up/down/left/
    right, mixing local 2D neighbours before the token-mix. x: (B, N, C)."""
    B, N, C = x.shape
    q = C // 4
    xx = x.transpose(1, 2).reshape(B, C, H, W)
    out = torch.zeros_like(xx)
    out[:, 0*q:1*q, :, :-1] = xx[:, 0*q:1*q, :, 1:]     # shift left  (look right)
    out[:, 1*q:2*q, :, 1:]  = xx[:, 1*q:2*q, :, :-1]    # shift right (look left)
    out[:, 2*q:3*q, :-1, :] = xx[:, 2*q:3*q, 1:, :]     # shift up    (look down)
    out[:, 3*q:4*q, 1:, :]  = xx[:, 3*q:4*q, :-1, :]    # shift down  (look up)
    if 4 * q < C:                                        # leftover channels unshifted
        out[:, 4*q:, :, :] = xx[:, 4*q:, :, :]
    return out.reshape(B, C, N).transpose(1, 2)


def _wkv_forward(w, u, k, v):
    """Numerically-stable causal WKV recurrence (RWKV-4). w,u:(C,) k,v:(B,L,C)."""
    B, L, C = k.shape
    a = torch.zeros(B, C, device=k.device, dtype=torch.float32)   # numerator state
    b = torch.zeros(B, C, device=k.device, dtype=torch.float32)   # denominator state
    p = torch.full((B, C), -1e30, device=k.device, dtype=torch.float32)
    kf, vf = k.float(), v.float()
    out = []
    for t in range(L):
        kt, vt = kf[:, t], vf[:, t]
        # output with bonus u
        q = torch.maximum(p, u + kt)
        e1, e2 = torch.exp(p - q), torch.exp(u + kt - q)
        out.append((e1 * a + e2 * vt) / (e1 * b + e2))
        # state update with decay w
        q2 = torch.maximum(p + w, kt)
        e1, e2 = torch.exp(p + w - q2), torch.exp(kt - q2)
        a, b, p = e1 * a + e2 * vt, e1 * b + e2, q2
    return torch.stack(out, dim=1).to(k.dtype)


def bi_wkv(w, u, k, v):
    """Bidirectional WKV: forward + reversed, averaged (image tokens are acausal)."""
    fwd = _wkv_forward(w, u, k, v)
    bwd = _wkv_forward(w, u, k.flip(1), v.flip(1)).flip(1)
    return 0.5 * (fwd + bwd)


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.time_decay = nn.Parameter(torch.zeros(dim))   # w = -exp(time_decay) < 0
        self.time_first = nn.Parameter(torch.zeros(dim))    # bonus u
        # per-branch token-shift mix coefficients
        self.mix_k = nn.Parameter(torch.ones(dim) * 0.5)
        self.mix_v = nn.Parameter(torch.ones(dim) * 0.5)
        self.mix_r = nn.Parameter(torch.ones(dim) * 0.5)

    def forward(self, x, H, W):
        xs = q_shift_2d(x, H, W)
        k = self.key(x * self.mix_k + xs * (1 - self.mix_k))
        v = self.value(x * self.mix_v + xs * (1 - self.mix_v))
        r = torch.sigmoid(self.receptance(x * self.mix_r + xs * (1 - self.mix_r)))
        w = -torch.exp(self.time_decay.float())
        wkv = bi_wkv(w, self.time_first.float(), k, v)
        return self.output(r * wkv)


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, dim, hidden_ratio=4):
        super().__init__()
        h = int(dim * hidden_ratio)
        self.key = nn.Linear(dim, h, bias=False)
        self.value = nn.Linear(h, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.mix_k = nn.Parameter(torch.ones(dim) * 0.5)
        self.mix_r = nn.Parameter(torch.ones(dim) * 0.5)

    def forward(self, x, H, W):
        xs = q_shift_2d(x, H, W)
        k = self.key(x * self.mix_k + xs * (1 - self.mix_k))
        k = torch.square(torch.relu(k))
        r = torch.sigmoid(self.receptance(x * self.mix_r + xs * (1 - self.mix_r)))
        return r * self.value(k)


class VRWKV_Block(nn.Module):
    def __init__(self, dim, hidden_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)
        self.att = VRWKV_SpatialMix(dim)
        self.ffn = VRWKV_ChannelMix(dim, hidden_ratio)

    def forward(self, x, H, W):
        x = x + self.att(self.ln1(x), H, W)
        x = x + self.ffn(self.ln2(x), H, W)
        return x


class VisionRWKV(nn.Module):
    def __init__(self, num_classes=3, img_size=224, patch_size=16,
                 dim=192, depth=12, hidden_ratio=4, in_chans=3):
        super().__init__()
        self.H = self.W = img_size // patch_size
        self.patch_embed = nn.Conv2d(in_chans, dim, patch_size, patch_size)
        n = self.H * self.W
        self.pos_embed = nn.Parameter(torch.zeros(1, n, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([VRWKV_Block(dim, hidden_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def get_classifier(self):
        return self.head

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            x = blk(x, self.H, self.W)
        return self.head(self.norm(x).mean(dim=1))


def build_vrwkv(num_classes=3, img_size=224, patch_size=16, dim=192, depth=12, **kw):
    return VisionRWKV(num_classes=num_classes, img_size=img_size,
                      patch_size=patch_size, dim=dim, depth=depth)


# ============================================================================
# registry + correctness helpers
# ============================================================================
BUILDERS = {
    "convnext": build_convnext,
    "mambad":   build_mambad,
    "vil":      build_vil,
    "vrwkv":    build_vrwkv,
}


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6   # millions


def overfit_one_batch(model, x, y, steps=60, lr=1e-3, device="cpu"):
    """Correctness gate: a well-wired model must drive a single batch to ~0 loss.
    Returns (start_loss, end_loss, end_acc)."""
    model = model.to(device).train()
    x, y = x.to(device), y.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    start = None
    for i in range(steps):
        opt.zero_grad()
        out = model(x)
        loss = lossf(out, y)
        loss.backward(); opt.step()
        if start is None:
            start = loss.item()
    end = loss.item()
    acc = (model(x).argmax(1) == y).float().mean().item()
    return start, end, acc
