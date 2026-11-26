"""
Conventional machine-learning baseline under the identical subject-level five-fold
splits used by the deep models. Each subject is described by intensity statistics, an
intensity histogram, gradient statistics, grey-level co-occurrence (Haralick) texture
and a local-binary-pattern histogram, averaged over its three hippocampal-band slices,
and classified with a gradient-boosted tree. The full metric suite (balanced accuracy,
sensitivity, specificity, macro-F1, macro-AUC) is reported as five-fold mean +/- SD.

Uses 09_train_cv.py only to locate slices and reproduce folds. Resumable: extracted
features are cached per cohort and reused on a re-run. Output: ml_baseline_summary.csv
"""
import importlib.util, os, glob, numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (balanced_accuracy_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score)
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def _find(n):
    for p in glob.glob(f"/kaggle/input/**/{n}", recursive=True)+[n, os.path.join(os.getcwd(),n)]:
        if p and os.path.exists(p): return p
    raise FileNotFoundError(n)
spec=importlib.util.spec_from_file_location("train_cv", _find("09_train_cv.py"))
T=importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
OUT=os.environ.get("CV_OUT") or ("/kaggle/working" if os.path.isdir("/kaggle") else ".")
LAB=list(range(T.N_CLASSES))

GL=32  # grey levels for GLCM
def slice_features(path):
    a=np.asarray(Image.open(path).convert("L"),dtype=np.float32)/255.0
    nz=a[a>0.02]
    if nz.size==0: nz=a.ravel()
    gx,gy=np.gradient(a); gm=np.sqrt(gx*gx+gy*gy)
    hist,_=np.histogram(nz,bins=16,range=(0,1),density=True)
    inten=[nz.mean(),nz.std(),np.median(nz),nz.min(),nz.max(),
           np.percentile(nz,10),np.percentile(nz,25),np.percentile(nz,75),np.percentile(nz,90),
           gm.mean(),gm.std(),float((a>0.5).mean()),float((a>0.02).mean())]
    q=(a*(GL-1)).astype(np.uint8)
    glcm=graycomatrix(q,distances=[1,2],angles=[0,np.pi/4,np.pi/2,3*np.pi/4],
                      levels=GL,symmetric=True,normed=True)
    tex=[graycoprops(glcm,p).mean() for p in
         ("contrast","dissimilarity","homogeneity","energy","correlation","ASM")]
    lbp=local_binary_pattern((a*255).astype(np.uint8),8,1,method="uniform")
    lh,_=np.histogram(lbp,bins=10,range=(0,10),density=True)
    return np.concatenate([inten,hist,tex,lh])

def subject_matrix(cohort,idx):
    cache=os.path.join(OUT,f"features_{cohort}.npz")
    if os.path.exists(cache):
        d=np.load(cache,allow_pickle=True); print(f"  [cache] loaded {cache}")
        return list(d["sids"]),d["X"],d["y"]
    rows={}
    for i,r in enumerate(idx.itertuples(),1):
        rows.setdefault(r.sid,[]).append(slice_features(r.path))
        if i%500==0: print(f"    ...{i}/{len(idx)} slices")
    sids=sorted(rows); X=np.stack([np.mean(rows[s],axis=0) for s in sids])
    y=idx.groupby("sid")["y"].first().reindex(sids).values.astype(int)
    np.savez(cache,sids=np.array(sids),X=X,y=y); print(f"  [cache] saved {cache}")
    return sids,X,y

def spec_macro(cm):
    tot=cm.sum(); s=[]
    for c in range(cm.shape[0]):
        tp=cm[c,c]; fp=cm[:,c].sum()-tp; fn=cm[c,:].sum()-tp; tn=tot-tp-fp-fn
        s.append(tn/(tn+fp) if (tn+fp) else 0.0)
    return float(np.mean(s))

results=[]
for cohort in ["adni","oasis600"]:
    idx,base=T.index_all(cohort)
    if idx.empty: print("skip",cohort); continue
    print(f"\n[{cohort}] features from {len(idx)} slices ...")
    sids,X,y=subject_matrix(cohort,idx)
    print(f"  {len(sids)} subjects, {X.shape[1]} features, classes={np.bincount(y)}")
    skf=StratifiedKFold(n_splits=T.N_FOLDS,shuffle=True,random_state=T.SEED)
    met={k:[] for k in ("balanced_accuracy","sensitivity_macro","specificity_macro","f1_macro","macro_auc")}
    for k,(tr,te) in enumerate(skf.split(X,y)):
        # shallow trees + L2 + balanced weights: a deliberately simple, well-regularised
        # reference, so any gap to the deep models is not from an over-tuned baseline
        clf=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.08,max_iter=500,
             l2_regularization=1.0,random_state=T.SEED,class_weight="balanced")
        clf.fit(X[tr],y[tr]); pred=clf.predict(X[te]); prob=clf.predict_proba(X[te])
        cm=confusion_matrix(y[te],pred,labels=LAB)
        met["balanced_accuracy"].append(balanced_accuracy_score(y[te],pred))
        met["sensitivity_macro"].append(recall_score(y[te],pred,labels=LAB,average="macro",zero_division=0))
        met["specificity_macro"].append(spec_macro(cm))
        met["f1_macro"].append(f1_score(y[te],pred,labels=LAB,average="macro",zero_division=0))
        try: met["macro_auc"].append(roc_auc_score(y[te],prob,multi_class="ovr",average="macro",labels=LAB))
        except Exception: met["macro_auc"].append(float("nan"))
        print(f"  fold {k+1}: bal_acc={met['balanced_accuracy'][-1]:.3f}")
    row=dict(cohort=cohort,model="GBM_structural")
    for k,v in met.items(): row[k]=f"{np.nanmean(v):.4f} ± {np.nanstd(v,ddof=1):.4f}"
    row["n_folds"]=T.N_FOLDS; results.append(row)
    print(f"  {cohort}: bal_acc={np.mean(met['balanced_accuracy']):.3f} +/- {np.std(met['balanced_accuracy'],ddof=1):.3f}")
pd.DataFrame(results).to_csv(os.path.join(OUT,"ml_baseline_summary.csv"),index=False)
print("\nSaved ml_baseline_summary.csv"); print(pd.DataFrame(results).to_string(index=False))
