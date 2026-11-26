"""
Within-cohort robustness experiments: multi-seed repeats and slice-count sensitivity.

Runs the four backbones within each cohort (its own train/val/test), looping over:
  - SEEDS: repeat each run with several random seeds and report mean +/- SD.
  - SLICE_COUNTS: use N slices per subject (1 = middle slice only, 3 = all three) to
    show the effect of the slice-count choice.

For every (cohort, model, seed, n_slices) it records subject-level metrics, writes
experiments_results_<TAG>.csv and prints mean +/- SD summaries. Resumable: any
(cohort, model, seed, n_slices) already in the CSV is skipped on a re-run. Keep
08_models.py beside this file. Vision-RWKV is the slowest model, so large grids are
best split across runs (trim MODELS/COHORTS; the resume guard handles the rest).
"""

import os, glob, random, time, json, importlib.util
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, precision_score,
    recall_score, matthews_corrcoef, cohen_kappa_score, log_loss)

# ============================ CONFIG ============================
MODELS_FILE = "08_models.py"
OUT      = "/kaggle/working" if os.path.isdir("/kaggle") else "./exp_out"
COHORTS  = ["adni", "oasis"]
MODELS   = ["convnext", "mambad", "vil", "vrwkv"]
SEEDS         = [1, 2]        # extra seeds (combine with the existing seed-42 runs)
SLICE_COUNTS  = [3]           # set to [1, 3] for the slice-count sensitivity
# TAG names the output CSV (experiments_results_<TAG>.csv) and the resume file.
# Use a different TAG per campaign so runs don't overwrite each other:
#   multi-seed (c)  -> "multiseed"   ;  slice-count (b) -> "slicecount"
TAG      = "multiseed"

IMG=224; BATCH=32; EPOCHS=80; PATIENCE=20; LR=1e-4
WARMUP_EPOCHS=3; BACKBONE_LR_MULT=0.1; LABEL_SMOOTH=0.1; WEIGHT_DECAY=0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CFG = {
    "convnext": dict(pretrained=True,  warmup=WARMUP_EPOCHS, lr=1e-4, batch=32),
    "vil":      dict(pretrained=False, warmup=0, lr=5e-4, batch=32, dim=192, depth=12, patch_size=16),
    "mambad":   dict(pretrained=False, warmup=0, lr=5e-4, batch=16, patch_size=32, depth=4, dim=192, d_state=16),
    "vrwkv":    dict(pretrained=False, warmup=0, lr=5e-4, batch=16, patch_size=32, depth=8, dim=192),
}
CLASSES=("CN","MCI","AD"); N_CLASSES=3; LABELS=[0,1,2]
# ===============================================================

os.makedirs(OUT, exist_ok=True)
print("device:", DEVICE, "| torch:", torch.__version__)

def _load_models():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for c in (os.path.join(here, MODELS_FILE), MODELS_FILE, os.path.join(os.getcwd(), MODELS_FILE)):
        if os.path.exists(c):
            spec = importlib.util.spec_from_file_location("models_p3", c)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    raise FileNotFoundError(MODELS_FILE)
M=_load_models(); BUILDERS=M.BUILDERS

def _find_slices(keyword):
    for base in ("/kaggle/input","/kaggle/working","."):
        if not os.path.isdir(base): continue
        for dp,dns,_ in os.walk(base):
            if os.path.basename(dp)=="slices" and os.path.isdir(os.path.join(dp,"train")) and keyword.lower() in dp.lower():
                return dp
    return None
DATA_DIRS={ds:_find_slices(ds) for ds in COHORTS}
print("slices dirs:", DATA_DIRS)

def index_slices(ds, n_slices):
    base=DATA_DIRS[ds]; rows=[]
    if not base: return pd.DataFrame(rows)
    # slice index k is encoded in the filename as _s{k}; middle slice is s1
    keep = {1:{1}, 2:{0,2}, 3:{0,1,2}}.get(n_slices, set(range(n_slices)))
    for split in ("train","val","test"):
        for yi,lab in enumerate(CLASSES):
            for ext in ("png","jpg","jpeg"):
                for p in glob.glob(os.path.join(base,split,lab,f"*.{ext}")):
                    fn=os.path.basename(p); sid=fn.rsplit("_s",1)[0]
                    try: k=int(fn.rsplit("_s",1)[1].split(".")[0])
                    except Exception: k=0
                    if k in keep:
                        rows.append(dict(split=split,label=lab,y=yi,path=p,sid=sid))
    return pd.DataFrame(rows)

MEAN,STD=[0.485,0.456,0.406],[0.229,0.224,0.225]
train_tf=transforms.Compose([transforms.Grayscale(3),transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(10),transforms.RandomAffine(0,translate=(0.05,0.05)),
  transforms.Resize((IMG,IMG)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
eval_tf=transforms.Compose([transforms.Grayscale(3),transforms.Resize((IMG,IMG)),
  transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
class DS(Dataset):
    def __init__(s,df,tf): s.df=df.reset_index(drop=True); s.tf=tf
    def __len__(s): return len(s.df)
    def __getitem__(s,i): r=s.df.iloc[i]; return s.tf(Image.open(r.path).convert("L")), int(r.y)
def loader(df,tf,sh,bs): return DataLoader(DS(df,tf),batch_size=bs,shuffle=sh,num_workers=2,pin_memory=(DEVICE=="cuda"))

def spec_macro(cm):
    out=[]; T=cm.sum()
    for c in range(3):
        tp=cm[c,c]; fp=cm[:,c].sum()-tp; fn=cm[c,:].sum()-tp; tn=T-tp-fp-fn
        out.append(tn/(tn+fp) if (tn+fp) else 0.0)
    return float(np.mean(out))
def metrics(y,pred,P):
    cm=confusion_matrix(y,pred,labels=LABELS); m=dict(
      accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred),
      precision_macro=precision_score(y,pred,labels=LABELS,average="macro",zero_division=0),
      recall_macro=recall_score(y,pred,labels=LABELS,average="macro",zero_division=0),
      specificity_macro=spec_macro(cm),
      f1_macro=f1_score(y,pred,labels=LABELS,average="macro",zero_division=0),
      mcc=matthews_corrcoef(y,pred),cohen_kappa=cohen_kappa_score(y,pred,labels=LABELS))
    try:
        if len(set(y))==N_CLASSES:
            m["macro_auc"]=roc_auc_score(y,P,multi_class="ovr",average="macro",labels=LABELS)
            m["pr_auc_macro"]=average_precision_score(np.eye(3)[y],P,average="macro")
        else: m["macro_auc"]=m["pr_auc_macro"]=float("nan")
        m["log_loss"]=log_loss(y,P,labels=LABELS)
    except Exception: m["macro_auc"]=m["pr_auc_macro"]=m["log_loss"]=float("nan")
    return m
def get_head(model):
    if hasattr(model,"get_classifier"):
        try:
            h=model.get_classifier()
            if isinstance(h,nn.Module): return h
        except Exception: pass
    for a in ("head","classifier","fc"):
        h=getattr(model,a,None)
        if isinstance(h,nn.Module): return h
    return None
def freeze(model,fr):
    h=get_head(model)
    if h is None:
        for p in model.parameters(): p.requires_grad=True
        return
    ids={id(p) for p in h.parameters()}
    for p in model.parameters(): p.requires_grad=(not fr) or (id(p) in ids)
@torch.no_grad()
def predict(model,df,bs):
    model.eval(); ps=[]; ys=[]
    for xb,yb in loader(df,eval_tf,False,bs):
        xb=xb.to(DEVICE)
        with torch.autocast(DEVICE,enabled=(DEVICE=="cuda")): p=torch.softmax(model(xb).float(),1)
        ps.append(p.cpu().numpy()); ys.append(yb.numpy())
    return np.concatenate(ps),np.concatenate(ys)
def subj(df,sp):
    d=df.reset_index(drop=True).copy(); d["p"]=list(sp)
    r=[(int(g["y"].iloc[0]),np.mean(np.stack(g["p"].values),0)) for _,g in d.groupby("sid")]
    return [x[0] for x in r], np.stack([x[1] for x in r])
def build(name):
    cfg={k:v for k,v in MODEL_CFG.get(name,{}).items() if k not in ("warmup","lr","batch")}
    return BUILDERS[name](num_classes=N_CLASSES,img_size=IMG,**cfg)

def run_one(name, idx, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cfg=MODEL_CFG.get(name,{}); bs=cfg.get("batch",BATCH); lr=cfg.get("lr",LR); wu=cfg.get("warmup",0)
    tr,va,te=(idx[idx.split==s] for s in ("train","val","test"))
    model=build(name).to(DEVICE)
    cnt=tr["y"].value_counts().reindex(LABELS).fillna(0).values
    w=torch.tensor(cnt.sum()/(N_CLASSES*np.clip(cnt,1,None)),dtype=torch.float32,device=DEVICE)
    lossf=nn.CrossEntropyLoss(weight=w,label_smoothing=LABEL_SMOOTH)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=WEIGHT_DECAY)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    if wu: freeze(model,True)
    best,bstate,bad=-1,None,0; tl=loader(tr,train_tf,True,bs); t0=time.time()
    for ep in range(EPOCHS):
        if wu and ep==wu:
            freeze(model,False)
            for g in opt.param_groups: g["lr"]=lr*BACKBONE_LR_MULT
        model.train()
        for xb,yb in tl:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
            with torch.autocast(DEVICE,enabled=(DEVICE=="cuda")): loss=lossf(model(xb),yb)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
        sch.step()
        vp,_=predict(model,va,bs); sy2,sp2=subj(va,vp)
        vscore=balanced_accuracy_score(sy2, sp2.argmax(1))   # subject-level val balanced accuracy
        if vscore>best: best,bstate,bad=vscore,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},0
        else:
            bad+=1
            if bad>=PATIENCE and ep>=wu: break
    if bstate: model.load_state_dict(bstate)
    tp,_=predict(model,te,bs); sy,sp=subj(te,tp)
    return metrics(sy,sp.argmax(1),sp), (time.time()-t0)/60

KEY=["cohort","model","seed","n_slices"]

def _key(ds,name,sd,nsl):
    # normalise types so CSV-loaded rows compare equal to freshly-made ones
    return (str(ds), str(name), int(sd), int(nsl))

def _load_done(out_path):
    """Return (existing_df, set_of_completed_keys). Survives a crashed run so a
    re-run resumes instead of starting over."""
    if not os.path.exists(out_path):
        return pd.DataFrame(), set()
    try:
        df=pd.read_csv(out_path)
    except Exception as e:
        print("!! could not read existing CSV, starting fresh:", e)
        return pd.DataFrame(), set()
    done={_key(r.cohort,r.model,r.seed,r.n_slices) for r in df.itertuples()
          if all(hasattr(r,k) for k in KEY)}
    return df, done

def main():
    out=os.path.join(OUT,f"experiments_results_{TAG}.csv")
    df, done = _load_done(out)
    if done:
        print(f"resume: found {len(done)} completed run(s) in {out} - these will be skipped")
    for ds in COHORTS:
        for nsl in SLICE_COUNTS:
            idx=None  # loaded lazily, only if there's unfinished work for this (ds,nsl)
            for name in MODELS:
                for sd in SEEDS:
                    if _key(ds,name,sd,nsl) in done:
                        print(f"-- skip (done): {ds} | {name} | seed {sd} | {nsl} slice(s)")
                        continue
                    if idx is None:
                        idx=index_slices(ds,nsl)
                    if idx.empty: print(f"!! no data {ds}"); break
                    print(f"\n== {ds} | {name} | seed {sd} | {nsl} slice(s) ==")
                    try:
                        m,mins=run_one(name,idx,sd)
                    except Exception as e:
                        import traceback; print("FAILED:",e); traceback.print_exc(); continue
                    row=dict(cohort=ds,model=name,seed=sd,n_slices=nsl,train_min=round(mins,2),**m)
                    df=pd.concat([df,pd.DataFrame([row])],ignore_index=True)
                    done.add(_key(ds,name,sd,nsl))
                    df.to_csv(out,index=False)   # save AFTER every run - crash-safe
                    print(f"   balacc={m['balanced_accuracy']:.3f} auc={m['macro_auc']:.3f} "
                          f"({mins:.1f} min)  [saved {len(df)} rows]")
    print("\n=== mean +/- SD over seeds (balanced accuracy) ===")
    if len(df):
        g=df.groupby(["cohort","n_slices","model"])["balanced_accuracy"].agg(["mean","std","count"])
        print(g.round(3).to_string())
    print("\nSaved:", out, "| total rows:", len(df))

if __name__=="__main__":
    main()
