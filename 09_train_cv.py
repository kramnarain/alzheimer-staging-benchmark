"""
Subject-level stratified 5-fold cross-validation trainer for both cohorts and all
four backbones (ConvNeXt, MambAD, Vision-xLSTM, Vision-RWKV).

  - Subject-level folds: every slice of a subject stays in one fold (no leakage).
  - Stratified by class so each fold keeps the CN/MCI/AD balance.
  - Per fold: that fold is the test set; a stratified 15% of the remaining subjects is
    the validation set (early stopping); the rest is training. Same on ADNI and OASIS-3.
  - Cross-cohort: after training on a cohort's fold, the model is also evaluated on the
    whole of the other cohort, giving within-cohort and transfer results in one run
    (ADNI->OASIS and OASIS->ADNI), each as a 5-fold mean +/- SD.
  - Resumable: one JSON per (cohort, model, fold) is written to the output folder and
    skipped if already present, so a run that times out can be re-run and continues.
  - Writes cv_summary_mean_sd.csv (mean +/- SD per model) and cv_folds_long.csv.

Run:  python 09_train_cv.py   (reads 08_models.py from the same folder; GPU recommended).
"""

import os, glob, random, time, json, importlib.util
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, precision_score,
    recall_score, matthews_corrcoef, cohen_kappa_score, log_loss)

# ============================ CONFIG ============================
MODELS_FILE = "08_models.py"
# OUT: Kaggle default, or set env CV_OUT (e.g. a Google Drive folder on Colab) so that
# per-fold results + preds persist across disconnects and the run resumes where it stopped.
OUT      = os.environ.get("CV_OUT") or ("/kaggle/working" if os.path.isdir("/kaggle") else "./cv_out")
# Extra place(s) to look for the slice datasets. On Colab set env CV_DATA to the folder
# that holds the dataset subfolders, so it isn't searched by walking all of /content.
_EXTRA_DATA = [b for b in [os.environ.get("CV_DATA"), "/content"] if b]
# Cohorts to run. ADNI plus the enlarged OASIS-3 (615, 205/class). The size-matched
# OASIS-3 (321) is available via COHORT_DIRS but shares subjects with the 615 set, so
# cross-cohort evaluation between the two OASIS variants is skipped (it would leak).
COHORTS  = ["adni", "oasis600"]
# Each cohort is located by a substring of its Kaggle dataset folder name (both OASIS
# datasets contain 'oasis', so we match the full dataset slug to tell them apart).
COHORT_DIRS = {
    "adni":     "ad-adni-3class-slices-v2",
    "oasis300": "oasis3-3class-slices-v2",   # original 321 (107/class), matched to ADNI
    "oasis600": "oasis3-slices-v2",          # enlarged 615 (205/class)
}
MODELS   = ["convnext", "mambad", "vil", "vrwkv"]
N_FOLDS  = 5            # 5-fold CV: enough folds for a mean +/- SD without starving the small cohorts
VAL_FRAC = 0.15         # held-out slice of the training folds used for early stopping
TAG      = "cv5xp"      # run tag: cross-cohort eval, also saves predictions and embeddings

IMG      = 224          # ImageNet input size, required by the pretrained ConvNeXt
EPOCHS   = 80           # upper bound; early stopping ends most runs well before this
PATIENCE = 20           # stop when validation balanced accuracy stalls for 20 epochs
LR       = 1e-4         # default AdamW rate (per-model rates override below)
WARMUP_EPOCHS    = 3    # ConvNeXt only: fit the head first, then unfreeze the backbone
BACKBONE_LR_MULT = 0.1  # lower rate for the pretrained backbone after warm-up, limits drift
LABEL_SMOOTH     = 0.1  # mild smoothing for the small, hard 3-class task
WEIGHT_DECAY     = 0.05 # AdamW decay, the ConvNeXt-style default
SEED     = 42
RUN_CHECKS = True
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ConvNeXt is ImageNet-pretrained, so it uses a low rate (1e-4) and a short head warm-up.
# The three from-scratch models use a higher rate (5e-4); MambAD and Vision-RWKV use
# batch 16 (vs 32) to fit the T4 memory budget. Depths/dims follow each model's reference.
MODEL_CFG = {
    "convnext": dict(pretrained=True,  warmup=WARMUP_EPOCHS, lr=1e-4, batch=32),
    "vil":      dict(pretrained=False, warmup=0, lr=5e-4, batch=32,
                     dim=192, depth=12, patch_size=16),
    "mambad":   dict(pretrained=False, warmup=0, lr=5e-4, batch=16,
                     patch_size=32, depth=4, dim=192, d_state=16),
    "vrwkv":    dict(pretrained=False, warmup=0, lr=5e-4, batch=16,
                     patch_size=32, depth=8, dim=192),
}
CLASSES   = ("CN", "MCI", "AD")
N_CLASSES = len(CLASSES)
LABELS    = list(range(N_CLASSES))
# ===============================================================

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(OUT, exist_ok=True)
PRED_DIR = os.path.join(OUT, "preds")   # per-fold predictions + embeddings for figures/stats
os.makedirs(PRED_DIR, exist_ok=True)
CKPT_DIR = os.path.join(OUT, "ckpts")   # trained model weights, so nothing ever needs retraining
os.makedirs(CKPT_DIR, exist_ok=True)
SAVE_CKPTS = True
print("device:", DEVICE, "| torch:", torch.__version__)


# ---- load model builders from 08_models.py ----
def _load_models():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    cands = [os.path.join(here, MODELS_FILE), MODELS_FILE, os.path.join(os.getcwd(), MODELS_FILE)]
    cands += glob.glob("/kaggle/input/**/08_models.py", recursive=True) if os.path.isdir("/kaggle") else []
    for cand in cands:
        if cand and os.path.exists(cand):
            spec = importlib.util.spec_from_file_location("models_p3", cand)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            print("models from:", cand)
            return mod
    raise FileNotFoundError(f"{MODELS_FILE} not found (put it next to this file or attach it).")

M = _load_models()
BUILDERS, count_params, overfit_one_batch = M.BUILDERS, M.count_params, M.overfit_one_batch


# ------------------------ locate + index slices ------------------------
def find_slices_dir(cohort):
    """Find the 'slices' folder for a cohort (basename 'slices', path contains the
    cohort's dataset slug from COHORT_DIRS, has a train/ subdir). Works on any mount."""
    key = COHORT_DIRS.get(cohort, cohort).lower()
    for base in _EXTRA_DATA + ["/kaggle/input", "/kaggle/working", "."]:
        if not os.path.isdir(base):
            continue
        for dp, dns, _ in os.walk(base):
            if (os.path.basename(dp) == "slices"
                    and key in dp.lower()
                    and os.path.isdir(os.path.join(dp, "train"))):
                return dp
    return None


def cross_ok(a, b):
    """True if a->b is a valid cross-cohort transfer. Two variants of the same cohort
    (e.g. oasis300 and oasis600) share subjects, so transfer between them is skipped."""
    base = lambda c: c.rstrip("0123456789")   # oasis300/oasis600 -> 'oasis'; adni -> 'adni'
    return base(a) != base(b)


def index_all(cohort):
    """Pool train/val/test into one slice index (CV reassigns the splits itself)."""
    base = find_slices_dir(cohort)
    if not base:
        return pd.DataFrame(), None
    rows = []
    for split in ("train", "val", "test"):
        for yi, lab in enumerate(CLASSES):
            for ext in ("png", "jpg", "jpeg"):
                for p in glob.glob(os.path.join(base, split, lab, f"*.{ext}")):
                    sid = os.path.basename(p).rsplit("_s", 1)[0]
                    rows.append(dict(cohort=cohort, label=lab, y=yi, path=p, sid=sid))
    return pd.DataFrame(rows), base


def subject_table(df):
    """One row per subject: sid + its label (slices of a subject share a label)."""
    return (df.groupby("sid")["y"].first().reset_index()
              .sort_values("sid").reset_index(drop=True))


# ------------------------ data / transforms ------------------------
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Grayscale(3), transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10), transforms.RandomAffine(0, translate=(0.05, 0.05)),
    transforms.Resize((IMG, IMG)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([
    transforms.Grayscale(3), transforms.Resize((IMG, IMG)),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class SliceDS(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True); self.tf = tf
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(r.path).convert("L")), int(r.y)


def loader(df, tf, shuffle, bs):
    return DataLoader(SliceDS(df, tf), batch_size=bs, shuffle=shuffle,
                      num_workers=2, pin_memory=(DEVICE == "cuda"))


# ------------------------ metrics ------------------------
def specificity_macro(cm):
    specs, total = [], cm.sum()
    for c in range(cm.shape[0]):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp; tn = total - tp - fp - fn
        specs.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return float(np.mean(specs))


def all_metrics(y_true, y_pred, y_prob):
    y_true, y_pred, y_prob = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    m = {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro":   precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "sensitivity_macro": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "specificity_macro": specificity_macro(cm),
        "f1_macro":          f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "mcc":               matthews_corrcoef(y_true, y_pred),
        "cohen_kappa":       cohen_kappa_score(y_true, y_pred, labels=LABELS),
    }
    try:
        if len(set(y_true)) == N_CLASSES:
            m["macro_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=LABELS)
            m["pr_auc_macro"] = average_precision_score(np.eye(N_CLASSES)[y_true], y_prob, average="macro")
        else:
            m["macro_auc"] = m["pr_auc_macro"] = float("nan")
        m["log_loss"] = log_loss(y_true, y_prob, labels=LABELS)
    except Exception:
        m["macro_auc"] = m["pr_auc_macro"] = m["log_loss"] = float("nan")
    m["confusion_matrix"] = cm.tolist()
    return m


# ------------------------ model helpers ------------------------
def get_head(model):
    if hasattr(model, "get_classifier"):
        try:
            h = model.get_classifier()
            if isinstance(h, nn.Module):
                return h
        except Exception:
            pass
    for attr in ("head", "classifier", "fc"):
        h = getattr(model, attr, None)
        if isinstance(h, nn.Module):
            return h
    return None


def freeze_backbone(model, freeze):
    head = get_head(model)
    if head is None:
        for p in model.parameters():
            p.requires_grad = True
        return
    head_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        p.requires_grad = (not freeze) or (id(p) in head_ids)


@torch.no_grad()
def predict(model, df, bs):
    model.eval()
    probs, ys = [], []
    for xb, yb in loader(df, eval_tf, False, bs):
        xb = xb.to(DEVICE)
        with torch.autocast(DEVICE, enabled=(DEVICE == "cuda")):
            p = torch.softmax(model(xb).float(), dim=1)
        probs.append(p.cpu().numpy()); ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def subject_level(df, slice_prob):
    d = df.reset_index(drop=True).copy()
    d["prob"] = list(slice_prob)
    rows = []
    for sid, g in d.groupby("sid"):
        mp = np.mean(np.stack(g["prob"].values), axis=0)
        rows.append((sid, int(g["y"].iloc[0]), mp))
    ys = [r[1] for r in rows]; ps = np.stack([r[2] for r in rows])
    return ys, ps.argmax(1), ps


def cost_metrics(model, bs):
    model.eval()
    x = torch.randn(bs, 3, IMG, IMG, device=DEVICE)
    if DEVICE == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(3):
            model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / 10
    peak = (torch.cuda.max_memory_allocated() / 1e6) if DEVICE == "cuda" else float("nan")
    return dict(params_M=count_params(model), latency_ms=dt / bs * 1000,
                throughput_ips=bs / dt, peak_mem_MB=peak)


def build(name):
    cfg = {k: v for k, v in MODEL_CFG.get(name, {}).items() if k not in ("warmup", "lr", "batch")}
    return BUILDERS[name](num_classes=N_CLASSES, img_size=IMG, **cfg)


def train_fold(name, tr, va, te):
    cfg = MODEL_CFG.get(name, {})
    bs = cfg.get("batch", 32); base_lr = cfg.get("lr", LR); warmup = cfg.get("warmup", 0)
    model = build(name).to(DEVICE)

    counts = tr["y"].value_counts().reindex(LABELS).fillna(0).values
    w = torch.tensor((counts.sum() / (N_CLASSES * np.clip(counts, 1, None))),
                     dtype=torch.float32, device=DEVICE)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTH)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    if warmup:
        freeze_backbone(model, True)

    best, best_state, bad = -1, None, 0
    tl = loader(tr, train_tf, True, bs)
    t0 = time.time(); epochs_run = 0
    for ep in range(EPOCHS):
        if warmup and ep == warmup:
            freeze_backbone(model, False)
            for gp in opt.param_groups:
                gp["lr"] = base_lr * BACKBONE_LR_MULT
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with torch.autocast(DEVICE, enabled=(DEVICE == "cuda")):
                loss = lossf(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        vp, _ = predict(model, va, bs)
        sy, spred, _ = subject_level(va, vp)
        vscore = balanced_accuracy_score(sy, spred)
        epochs_run = ep + 1
        if vscore > best:
            best, best_state, bad = vscore, {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE and ep >= warmup:
                break
    train_seconds = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)

    cm = cost_metrics(model, bs)
    timing = dict(train_minutes=round(train_seconds / 60, 2), epochs_trained=epochs_run,
                  sec_per_epoch=round(train_seconds / max(epochs_run, 1), 1))
    # the caller evaluates the model (within-cohort test fold AND every other cohort)
    return model, cm, timing, bs


def eval_on(model, idx_df, bs):
    """Evaluate a trained model on any slice index. Returns (metrics, per-slice preds).
    The per-slice prediction frame (sid, slice, y, prob_*) lets us regenerate ROC curves,
    McNemar/DeLong tests, bootstrap CIs and the 1-vs-3 slice ablation after the run."""
    probs, ys = predict(model, idx_df, bs)
    sl = all_metrics(ys, probs.argmax(1), probs)
    sy, spred, sprob = subject_level(idx_df, probs)
    su = all_metrics(sy, spred, sprob)
    d = idx_df.reset_index(drop=True)[["sid", "y", "path"]].copy()
    d["slice"] = d["path"].apply(os.path.basename)
    for i, c in enumerate(CLASSES):
        d[f"prob_{c}"] = probs[:, i]
    d = d.drop(columns=["path"])
    return {"slice": sl, "subject": su}, d


def to_subject_preds(slice_df):
    """Collapse per-slice predictions to one row per subject (mean of slice probabilities)."""
    pc = [f"prob_{c}" for c in CLASSES]
    g = slice_df.groupby("sid")
    out = g[pc].mean()
    out["y"] = g["y"].first()
    return out.reset_index()


@torch.no_grad()
def extract_embeddings(model, idx_df, bs):
    """Penultimate-layer embeddings (the input to the classifier head) for t-SNE, averaged
    per subject. Architecture-agnostic: a forward pre-hook on the head captures its input."""
    head = get_head(model)
    if head is None:
        return None
    buf = []
    def _hook(_m, inp):
        buf.append(inp[0].detach().float().reshape(inp[0].shape[0], -1).cpu().numpy())
    h = head.register_forward_pre_hook(_hook)
    try:
        predict(model, idx_df, bs)   # runs the model; the hook collects head inputs
    finally:
        h.remove()
    F = np.concatenate(buf, axis=0)
    d = idx_df.reset_index(drop=True)[["sid", "y"]].copy()
    emb = pd.DataFrame(F, columns=[f"f{i}" for i in range(F.shape[1])])
    out = pd.concat([d, emb], axis=1).groupby("sid").mean(numeric_only=True).reset_index()
    return out


# ------------------------ resume bookkeeping ------------------------
def fold_path(cohort, name, k):
    return os.path.join(OUT, f"fold_{cohort}_{name}_f{k}_{TAG}.json")


def already_done(cohort, name, k):
    """Done if the JSON exists in OUT, or anywhere under /kaggle/input (prev version)."""
    if os.path.exists(fold_path(cohort, name, k)):
        return fold_path(cohort, name, k)
    fn = os.path.basename(fold_path(cohort, name, k))
    if os.path.isdir("/kaggle/input"):
        hits = glob.glob(f"/kaggle/input/**/{fn}", recursive=True)
        if hits:
            return hits[0]
    return None


# ------------------------ correctness gate ------------------------
def run_checks():
    print("\n=== correctness checks (param count + overfit-one-batch) ===")
    xb = torch.randn(8, 3, IMG, IMG); yb = torch.randint(0, N_CLASSES, (8,))
    GATE_LR = {"convnext": 1e-3, "vil": 1e-3, "mambad": 5e-4, "vrwkv": 5e-4}
    for name in MODELS:
        try:
            model = build(name); p = count_params(model)
            s, e, acc = overfit_one_batch(model, xb, yb, steps=120,
                                          lr=GATE_LR.get(name, 1e-3), device=DEVICE)
            flag = "OK" if (e < s * 0.5 and acc > 0.9) else "CHECK"
            print(f"  {name:9s} params={p:6.2f}M  loss {s:.3f}->{e:.3f}  acc={acc:.2f}  [{flag}]")
        except Exception as ex:
            print(f"  {name:9s} FAILED: {ex}")
    print("=== end checks ===\n")


# ------------------------ aggregate ------------------------
def aggregate():
    rows = []
    def _load(f):
        d = json.load(open(f))
        return d if isinstance(d, list) else [d]
    for f in glob.glob(os.path.join(OUT, f"fold_*_{TAG}.json")):
        rows.extend(_load(f))
    # also pull any folds that live only in an attached previous version
    KEY = ["train_cohort", "test_cohort", "model", "fold", "level"]
    def _key(r):
        return tuple(r[c] for c in KEY)
    if os.path.isdir("/kaggle/input"):
        seen = {_key(r) for r in rows}
        for f in glob.glob(f"/kaggle/input/**/fold_*_{TAG}.json", recursive=True):
            for r in _load(f):
                if _key(r) not in seen:
                    rows.append(r); seen.add(_key(r))
    if not rows:
        print("no fold results yet."); return
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(OUT, f"cv_folds_long_{TAG}.csv"), index=False)
    metric_cols = ["accuracy", "balanced_accuracy", "precision_macro", "sensitivity_macro",
                   "specificity_macro", "f1_macro", "mcc", "cohen_kappa", "macro_auc",
                   "pr_auc_macro", "log_loss"]
    g = long.groupby(["train_cohort", "test_cohort", "eval", "model", "level"])
    mean = g[metric_cols].mean().round(4)
    sd = g[metric_cols].std(ddof=1).round(4)
    out = mean.copy()
    for c in metric_cols:
        out[c] = mean[c].astype(str) + " ± " + sd[c].astype(str)
    out["n_folds"] = g.size()
    out.reset_index().to_csv(os.path.join(OUT, f"cv_summary_mean_sd_{TAG}.csv"), index=False)
    print("\n=== 5-fold subject-level balanced accuracy (mean ± sd) ===")
    sub = out.reset_index()
    sub = sub[sub.level == "subject"][["train_cohort", "test_cohort", "eval", "model",
                                       "balanced_accuracy", "sensitivity_macro",
                                       "specificity_macro", "n_folds"]]
    print(sub.to_string(index=False))
    print("\nSaved: cv_summary_mean_sd_%s.csv  and  cv_folds_long_%s.csv" % (TAG, TAG))


# ------------------------ main ------------------------
def _rec(train_c, test_c, evaltype, name, k, lv, mm, cm, timing):
    return dict(train_cohort=train_c, test_cohort=test_c, eval=evaltype,
                model=name, fold=k, level=lv,
                **{kk: vv for kk, vv in mm.items() if kk != "confusion_matrix"},
                confusion_matrix=json.dumps(mm["confusion_matrix"]), **cm, **timing)


def main():
    if RUN_CHECKS:
        run_checks()
    # index every cohort up front -- the other cohort is needed for cross-cohort eval
    IDX, SUBJ, FOLDS = {}, {}, {}
    for c in COHORTS:
        d, base = index_all(c)
        if d.empty:
            print(f"!! no slices found for '{c}', skipping"); continue
        IDX[c] = d; SUBJ[c] = subject_table(d)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        FOLDS[c] = list(skf.split(SUBJ[c]["sid"], SUBJ[c]["y"]))
        print(f"\n########## {c}: {len(d)} slices, {len(SUBJ[c])} subjects "
              f"{dict(SUBJ[c]['y'].value_counts().reindex(LABELS))} | {base}")
    cohorts = list(IDX.keys())

    run_t0 = time.time()
    for cohort in cohorts:
        idx, subj, folds = IDX[cohort], SUBJ[cohort], FOLDS[cohort]
        for name in MODELS:
            for k, (trv_i, te_i) in enumerate(folds):
                hit = already_done(cohort, name, k)
                if hit:
                    print(f"[skip] {cohort}/{name} fold {k} (found {os.path.basename(hit)})")
                    continue
                te_sids = subj.iloc[te_i]["sid"].values
                trv = subj.iloc[trv_i]
                tr_sids, va_sids = train_test_split(
                    trv["sid"].values, test_size=VAL_FRAC, random_state=SEED,
                    stratify=trv["y"].values)
                tr = idx[idx.sid.isin(tr_sids)]; va = idx[idx.sid.isin(va_sids)]
                te = idx[idx.sid.isin(te_sids)]
                print(f"\n===== {cohort}/{name} fold {k+1}/{N_FOLDS} "
                      f"(train {len(tr_sids)} / val {len(va_sids)} / test {len(te_sids)} subj) =====")
                try:
                    model, cm, timing, bs = train_fold(name, tr, va, te)
                except Exception as ex:
                    import traceback; print(f"!! FAILED {cohort}/{name} f{k}: {ex}")
                    traceback.print_exc(); continue

                recs = []
                # within-cohort: the held-out test fold (save per-slice preds for ROC /
                # significance / the 1-vs-3-slice ablation)
                wm, wpred = eval_on(model, te, bs)
                for lv in ("slice", "subject"):
                    recs.append(_rec(cohort, cohort, "within", name, k, lv, wm[lv], cm, timing))
                wpred.insert(0, "eval", "within")
                wpred.to_csv(os.path.join(PRED_DIR,
                             f"pred_{cohort}_to_{cohort}_{name}_f{k}_{TAG}.csv"), index=False)
                # cross-cohort: the whole of every other cohort (never seen in training)
                cross_disp = {}
                for other in cohorts:
                    if other == cohort or not cross_ok(cohort, other):
                        continue
                    xm, xpred = eval_on(model, IDX[other], bs)
                    for lv in ("slice", "subject"):
                        recs.append(_rec(cohort, other, "cross", name, k, lv, xm[lv], cm, timing))
                    to_subject_preds(xpred).to_csv(os.path.join(PRED_DIR,
                        f"pred_{cohort}_to_{other}_{name}_f{k}_{TAG}.csv"), index=False)
                    cross_disp[other] = xm["subject"]["balanced_accuracy"]

                # fold 0 only: penultimate embeddings of EVERY subject in BOTH cohorts,
                # from this cohort's model -> the t-SNE domain-shift figures
                if k == 0:
                    for ec in cohorts:
                        if ec != cohort and not cross_ok(cohort, ec):
                            continue
                        emb = extract_embeddings(model, IDX[ec], bs)
                        if emb is not None:
                            emb.insert(0, "cohort", ec)
                            emb.to_csv(os.path.join(PRED_DIR,
                                f"emb_trainedON_{cohort}_{name}_subjectsOF_{ec}_{TAG}.csv"),
                                index=False)

                if SAVE_CKPTS:   # save weights so any future figure/metric needs only inference
                    torch.save(model.state_dict(),
                               os.path.join(CKPT_DIR, f"ckpt_{cohort}_{name}_f{k}_{TAG}.pt"))

                json.dump(recs, open(fold_path(cohort, name, k), "w"))
                del model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                print(f"  fold {k+1}: within {cohort} bal_acc={wm['subject']['balanced_accuracy']:.3f}"
                      + "".join(f" | {cohort}->{o} {v:.3f}" for o, v in cross_disp.items())
                      + f"  ({timing['train_minutes']:.1f} min)")
                aggregate()  # refresh tables after every fold (keeps partial results usable)
    aggregate()
    print(f"\nTOTAL wall-clock this run: {(time.time()-run_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
