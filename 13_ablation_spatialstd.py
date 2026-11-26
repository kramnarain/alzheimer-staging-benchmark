"""
Spatial-standardisation ablation on the OASIS-3 (615) cohort. Trains each of the four
models on the registered pipeline and on the light (unregistered, no skull-strip)
pipeline, single seed 42, three slices, base configuration, and compares subject-level
balanced accuracy. Reuses 11_experiments.py for the training and metrics.

Attach the registered and light OASIS-3 slice datasets plus the code
(08_models.py + 11_experiments.py). The two sets share subjects and splits, so the
comparison is like-for-like. Resumable: results are written to spatial_std_615.csv
after each run and skipped if already present.
"""
import importlib.util, os, glob, pandas as pd
def _find(n):
    for p in glob.glob(f"/kaggle/input/**/{n}", recursive=True)+[n, os.path.join(os.getcwd(),n)]:
        if p and os.path.exists(p): return p
    raise FileNotFoundError(n)
spec=importlib.util.spec_from_file_location("exp12", _find("11_experiments.py"))
E=importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
OUT=E.OUT; MODELS=["convnext","mambad","vil","vrwkv"]; SEED=42
def find_slices(slug):
    for base in ("/kaggle/input","/kaggle/working","."):
        if not os.path.isdir(base): continue
        for dp,_,_ in os.walk(base):
            if os.path.basename(dp)=="slices" and slug in dp.lower() and os.path.isdir(os.path.join(dp,"train")):
                return dp
    return None
DIRS={"registered":find_slices("oasis3-slices-v2"),"light":find_slices("oasis3-slices-light-v2")}
print("pipeline dirs:",DIRS)
assert DIRS["registered"] and DIRS["light"], "attach both oasis3-slices-v2 and oasis3-slices-light-v2"
out=os.path.join(OUT,"spatial_std_615.csv")
done=set(); rows=[]
if os.path.exists(out):
    prev=pd.read_csv(out); rows=prev.to_dict("records")
    done={(r["pipeline"],r["model"]) for r in rows}
    print("resume: already have",done)
for ver,base in DIRS.items():
    E.DATA_DIRS["oasis"]=base
    idx=E.index_slices("oasis",3)
    print(f"\n[{ver}] {len(idx)} slices, {idx['sid'].nunique()} subjects | {base}")
    for name in MODELS:
        if (ver,name) in done:
            print(f"  skip {ver}/{name} (done)"); continue
        print(f"  == {ver} / {name} ==")
        m,mins=E.run_one(name,idx,SEED)
        rows.append(dict(pipeline=ver,model=name,balanced_accuracy=round(m["balanced_accuracy"],4),
                         macro_auc=round(m["macro_auc"],4),train_min=round(mins,2)))
        pd.DataFrame(rows).to_csv(out,index=False)
        print(f"     balacc={m['balanced_accuracy']:.3f} auc={m['macro_auc']:.3f} ({mins:.1f} min)")
print("\n=== spatial standardisation (OASIS-3, 615): balanced accuracy ===")
d=pd.DataFrame(rows)
piv=d.pivot(index="model",columns="pipeline",values="balanced_accuracy")
print(piv.to_string()); print("\nSaved",out)
