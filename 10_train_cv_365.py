"""
One-year-window sensitivity run.

Re-runs the exact 5-fold protocol of 09_train_cv.py with one change: OASIS-3 is
relabelled using a tighter CDR-to-scan pairing window of 365 days (one year) instead
of 730 days. ADNI is unchanged, so any change in the result is attributable only to
the stricter temporal pairing.

oasis365_labels.csv (built from the CDR and MR-session tables) lists the resulting
balanced 165/class = 495-subject cohort and each subject's 365-day label. This wrapper
imports 09_train_cv.py as a module, filters OASIS-3 to those 495 subjects with their
365-day labels, and runs the same main() under a distinct tag (cv5xp365) so results do
not collide with the 730-day run and resume works independently.

Run:  python 10_train_cv_365.py   (needs 09_train_cv.py, 08_models.py, oasis365_labels.csv).
"""

import importlib.util, os, glob
import pandas as pd


def _find(name):
    cands = glob.glob(f"/kaggle/input/**/{name}", recursive=True) + [name, os.path.join(os.getcwd(), name)]
    for p in cands:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found (attach it as an input).")


# ---- import the original trainer WITHOUT running it ----
_src = _find("09_train_cv.py")
_spec = importlib.util.spec_from_file_location("train_cv", _src)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
print("protocol from:", _src)

# ---- load the 365-day OASIS-3 label map (subject_id -> label365) ----
_L = pd.read_csv(_find("oasis365_labels.csv"))
OA365 = dict(zip(_L["subject_id"].astype(str), _L["label365"].astype(str)))
print(f"365-day OASIS-3 cohort: {len(OA365)} subjects",
      _L["label365"].value_counts().to_dict(),
      f"| relabelled vs 730d: {(_L['label365'] != _L['label730']).sum()}")

_Y = {c: i for i, c in enumerate(T.CLASSES)}
_orig_index_all = T.index_all


def index_all_365(cohort):
    """Original indexer, then for OASIS-3 keep only the 495 one-year-window
    subjects and override each slice's label from the 365-day pairing.
    ADNI is returned untouched."""
    df, base = _orig_index_all(cohort)
    if df.empty or not cohort.startswith("oasis"):
        return df, base
    df = df[df["sid"].isin(OA365)].copy()
    df["label"] = df["sid"].map(OA365)
    df["y"] = df["label"].map(_Y)
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    df["y"] = df["y"].astype(int)
    return df, base


T.index_all = index_all_365

# ---- distinct run so it never collides with the 730-day CV ----
T.TAG = "cv5xp365"
T.COHORTS = ["adni", "oasis600"]   # ADNI (unchanged) + the 365-day OASIS-3
T.MODELS = ["convnext", "mambad", "vil", "vrwkv"]
# Do not save per-fold weights: 40 checkpoints overflow a ~20 GB working disk, and
# all downstream analysis is rebuilt from the small per-slice prediction CSVs in preds/.
T.SAVE_CKPTS = False

if __name__ == "__main__":
    T.main()
