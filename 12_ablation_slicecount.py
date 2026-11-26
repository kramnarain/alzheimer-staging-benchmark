"""
Slice-count ablation on the OASIS-3 (615) and ADNI (292) cohorts. Reuses
11_experiments.py to train each model with one slice (middle only) and three slices
(all three) per subject, from the existing registered slices (no re-preprocessing).
Single seed 42, base configuration.
Attach the two registered slice datasets plus the code (08_models.py + 11_experiments.py).
Output: experiments_results_slicecount615.csv
"""
import importlib.util, os, glob
def _find(n):
    for p in glob.glob(f"/kaggle/input/**/{n}", recursive=True)+[n, os.path.join(os.getcwd(),n)]:
        if p and os.path.exists(p): return p
    raise FileNotFoundError(n)
spec=importlib.util.spec_from_file_location("exp12", _find("11_experiments.py"))
E=importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
# force the REGISTERED dirs by exact slug, so an attached light dataset is never picked
def _slug(slug):
    import os
    for base in ("/kaggle/input","/kaggle/working","."):
        if not os.path.isdir(base): continue
        for dp,_,_ in os.walk(base):
            if os.path.basename(dp)=="slices" and slug in dp.lower() and os.path.isdir(os.path.join(dp,"train")):
                return dp
    return None
for ds,slug in (("adni","ad-adni-3class-slices-v2"),("oasis","oasis3-slices-v2")):
    d=_slug(slug)
    if d: E.DATA_DIRS[ds]=d
print("slice-count dirs:",E.DATA_DIRS)
E.SLICE_COUNTS=[1,3]
E.SEEDS=[42]
E.MODELS=["convnext","mambad","vil","vrwkv"]
E.TAG="slicecount615"
if __name__=="__main__":
    E.main()
