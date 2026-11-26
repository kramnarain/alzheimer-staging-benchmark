"""
04b_clean_light_oasis.py -- generate the LIGHT (unregistered) OASIS-3 615 slices
for the spatial-standardisation ablation re-run on the enlarged cohort.

It reuses 04_clean_and_extract.py unchanged (imported as a module, so its
__main__ block does NOT run), and only overrides the config so the pipeline
omits MNI registration and skull-stripping. N4 bias correction and the
hippocampal-band slice extraction are kept, which matches the "baseline /
simplified" pipeline the ablation compares against the registered one. The same
615-subject inventory and SEED are used, so the subject-level train/val/test
split is identical to the registered run and the comparison is like-for-like.

CPU is capped to a few threads so your other work and the VM keep their cores.
Output goes to a NEW folder (processed_light) so it never touches the registered
processed_clean set. Resumable (per-subject done-markers).

Run locally:  python 04b_clean_light_oasis.py
Then upload  <processed_light>\oasis\slices  as a Kaggle dataset named
'oasis3-slices-light-v2'.
"""
import os
# ---- CPU politeness: set BEFORE heavy libraries initialise ----
N_THREADS = 2          # raise for speed, lower to 1 to be even gentler
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
    os.environ[_v] = str(N_THREADS)
import importlib.util
import SimpleITK as sitk; sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(N_THREADS)
try:
    import ctypes  # below-normal process priority on Windows
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
except Exception: pass

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("clean04", os.path.join(HERE, "04_clean_and_extract.py"))
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)   # does NOT run main

# ---- LIGHT pipeline overrides (OASIS-3 only) ----
C.CLEAN_THREADS = N_THREADS
C.DO_N4        = True        # keep bias-field correction
C.DO_REGISTER  = False       # NO MNI registration
C.DO_SKULLSTRIP= False       # NO skull-stripping
C.PREPROC_VER  = "light_v2"  # distinct done-markers
C.OUT = "data/processed_light"
C.CKPT = os.path.join(C.OUT, "_clean_ckpt"); os.makedirs(C.CKPT, exist_ok=True)
# use the remapped inventory (raw scans were moved into an Oasis3v2 parent folder)
_remap=os.path.join(os.path.dirname(C.OASIS_CSV),"oasis_inventory_remap.csv")
if os.path.exists(_remap):
    C.OASIS_CSV=_remap; print("using remapped inventory:",_remap)
try: C.torch.set_num_threads(N_THREADS)
except Exception: pass

print(f"LIGHT OASIS-3 preprocessing -> {C.OUT}  (register={C.DO_REGISTER}, skullstrip={C.DO_SKULLSTRIP}, threads={N_THREADS})")
C.main()
