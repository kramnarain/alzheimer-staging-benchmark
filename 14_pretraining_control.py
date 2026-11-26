"""
Pretraining control. Re-runs the identical five-fold protocol of 09_train_cv.py for
ConvNeXt with ImageNet pretraining turned off (trained from scratch, no head warm-up,
the from-scratch learning rate), on ADNI and OASIS-3. Compared with the pretrained
ConvNeXt, this isolates how much of its advantage is pretraining rather than
architecture. Tag 'cv5scratch'.
"""
import importlib.util, os, glob
def _find(n):
    for p in glob.glob(f"/kaggle/input/**/{n}", recursive=True)+[n, os.path.join(os.getcwd(),n)]:
        if p and os.path.exists(p): return p
    raise FileNotFoundError(n)
spec=importlib.util.spec_from_file_location("train_cv", _find("09_train_cv.py"))
T=importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
print("protocol from 09_train_cv.py")
T.MODELS=["convnext"]
# from scratch: no ImageNet weights, no warm-up, the from-scratch LR used by the other nets
T.MODEL_CFG["convnext"]=dict(pretrained=False, warmup=0, lr=5e-4, batch=32)
T.TAG="cv5scratch"
T.COHORTS=["adni","oasis600"]
T.SAVE_CKPTS=False
if __name__=="__main__":
    T.main()
