"""
Additional ablations for the reference model ConvNeXt, on both cohorts, under the same
five-fold protocol. Varies: (a) training-set size (50/75/100 percent of the training
folds), (b) learning rate (x0.5, base, x2), (c) batch size (16/32/64) and (d) data
augmentation (on/off). Within-cohort subject-level balanced accuracy is reported as
five-fold mean +/- SD per configuration.

Reuses 09_train_cv.py (index_all, subject_table, train_fold, eval_on). Resumable: one
JSON per (cohort, configuration, fold) is written and skipped if already present, so a
timed-out run can be re-run and continues. Output: ablations_summary.csv and
ablations_folds_long.csv
"""
import importlib.util, os, glob, json, re, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
def _find(n):
    for p in glob.glob(f"/kaggle/input/**/{n}", recursive=True)+[n, os.path.join(os.getcwd(),n)]:
        if p and os.path.exists(p): return p
    raise FileNotFoundError(n)
spec=importlib.util.spec_from_file_location("train_cv", _find("09_train_cv.py"))
T=importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
OUT=os.environ.get("CV_OUT") or ("/kaggle/working" if os.path.isdir("/kaggle") else ".")
os.makedirs(OUT,exist_ok=True)
_orig_tf=T.train_tf; NOAUG=T.eval_tf
COHORTS=["adni","oasis600"]
BASE=dict(pretrained=True,warmup=3,lr=1e-4,batch=32)
CONFIGS=[
 ("baseline_frac100_lr1e-4_bs32_augon", dict(BASE), 1.0, True),
 ("trainsize_75",  dict(BASE), 0.75, True),
 ("trainsize_50",  dict(BASE), 0.50, True),
 ("lr_x0.5_5e-5",  dict(BASE,lr=5e-5), 1.0, True),
 ("lr_x2_2e-4",    dict(BASE,lr=2e-4), 1.0, True),
 ("batch_16",      dict(BASE,batch=16), 1.0, True),
 ("batch_64",      dict(BASE,batch=64), 1.0, True),
 ("aug_off",       dict(BASE), 1.0, False),
]
def fp(cohort,cfg,k): return os.path.join(OUT,f"abl_{cohort}_{cfg}_f{k}.json")
def done(cohort,cfg,k):
    if os.path.exists(fp(cohort,cfg,k)): return fp(cohort,cfg,k)
    hits=glob.glob(f"/kaggle/input/**/abl_{cohort}_{cfg}_f{k}.json",recursive=True) if os.path.isdir("/kaggle/input") else []
    return hits[0] if hits else None
def subsample(sids,ys,frac,seed=T.SEED):
    if frac>=1.0: return sids
    rng=np.random.RandomState(seed); keep=[]
    for c in np.unique(ys):
        cs=sids[ys==c]; n=max(1,int(round(len(cs)*frac)))
        keep+=list(rng.choice(cs,n,replace=False))
    return np.array(keep)
def aggregate():
    rows=[]
    seen=set()
    for f in glob.glob(os.path.join(OUT,"abl_*.json"))+ (glob.glob("/kaggle/input/**/abl_*.json",recursive=True) if os.path.isdir("/kaggle/input") else []):
        r=json.load(open(f)); key=(r["cohort"],r["configuration"],r["fold"])
        if key in seen: continue
        seen.add(key); rows.append(r)
    if not rows: return
    long=pd.DataFrame(rows); long.to_csv(os.path.join(OUT,"ablations_folds_long.csv"),index=False)
    g=long.groupby(["cohort","configuration"])["balanced_accuracy"]
    summ=g.agg(["mean","std","count"]).reset_index()
    summ["balanced_accuracy"]=summ["mean"].round(4).astype(str)+" ± "+summ["std"].round(4).astype(str)
    summ=summ.rename(columns={"count":"n_folds"})[["cohort","configuration","balanced_accuracy","n_folds"]]
    # keep config order
    order={c[0]:i for i,c in enumerate(CONFIGS)}
    summ["_o"]=summ["configuration"].map(order); summ=summ.sort_values(["cohort","_o"]).drop(columns="_o")
    summ.to_csv(os.path.join(OUT,"ablations_summary.csv"),index=False)

for cohort in COHORTS:
    idx,base=T.index_all(cohort)
    if idx.empty: print("skip",cohort); continue
    subj=T.subject_table(idx)
    FOLDS=list(StratifiedKFold(n_splits=T.N_FOLDS,shuffle=True,random_state=T.SEED).split(subj["sid"],subj["y"]))
    print(f"\n########## {cohort}: {len(idx)} slices, {len(subj)} subjects | {base}")
    for name,cfg,frac,aug in CONFIGS:
        print(f"\n=== {cohort} / {name} ===")
        for k,(trv_i,te_i) in enumerate(FOLDS):
            hit=done(cohort,name,k)
            if hit: print(f"  [skip] fold {k+1} ({os.path.basename(hit)})"); continue
            te_sids=subj.iloc[te_i]["sid"].values; trv=subj.iloc[trv_i]
            tr_sids,va_sids=train_test_split(trv["sid"].values,test_size=T.VAL_FRAC,
                random_state=T.SEED,stratify=trv["y"].values)
            if frac<1.0:
                tr_y=subj.set_index("sid").loc[tr_sids,"y"].values
                tr_sids=subsample(tr_sids,tr_y,frac)
            tr=idx[idx.sid.isin(tr_sids)]; va=idx[idx.sid.isin(va_sids)]; te=idx[idx.sid.isin(te_sids)]
            T.MODEL_CFG["convnext"]=dict(cfg); T.train_tf=_orig_tf if aug else NOAUG
            try:
                model,cm,timing,bs=T.train_fold("convnext",tr,va,te)
                m,_=T.eval_on(model,te,bs); ba=float(m["subject"]["balanced_accuracy"])
                del model
                import torch
                if T.DEVICE=="cuda": torch.cuda.empty_cache()
            finally:
                T.train_tf=_orig_tf
            json.dump(dict(cohort=cohort,configuration=name,fold=k,balanced_accuracy=ba,
                           n_train=int(len(tr_sids))), open(fp(cohort,name,k),"w"))
            print(f"  fold {k+1}: bal_acc={ba:.3f}  (n_train={len(tr_sids)})")
            aggregate()
aggregate()
print("\nSaved ablations_summary.csv")
if os.path.exists(os.path.join(OUT,"ablations_summary.csv")):
    print(pd.read_csv(os.path.join(OUT,"ablations_summary.csv")).to_string(index=False))
