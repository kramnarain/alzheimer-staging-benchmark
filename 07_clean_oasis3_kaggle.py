"""
OASIS-3 label + clean, on Kaggle (produces the 2D slices for training).

Runs on the uploaded raw dataset (oasis3-t1w-raw): the T1w NIfTIs in oasis3_t1w/
plus OASIS3_UDSb4_cdr.csv. CDR labelling uses nearest-assessment matching, and the
imaging follows the same steps as 04_clean_and_extract.py (N4 bias correction,
hippocampal-band 224x224 coronal slices, subject-level 70/15/15 splits), the same
preprocessing ADNI gets, so the cohorts are comparable.

Output (to /kaggle/working):
    oasis/slices/{train,val,test}/{CN,MCI,AD}/<subject>_s<k>.png
    oasis/volumes/<subject>.nii.gz
    oasis_inventory.csv , oasis_manifest.csv

Then zip oasis/slices and use it exactly like the ADNI slices in the trainer
(add "oasis" to DATASETS).

PREPROCESSING v2 (registration + skull-stripping):
    Every volume is now affine-registered to the MNI152 template and the skull is
    removed with the template brain mask BEFORE slicing. This makes the fixed slice
    fractions (0.45/0.50/0.55) hit the SAME anatomy in every subject (previously they
    did not, because brains differ in size/position), and removes non-brain pixels.
    ADNI must be re-cleaned the same way (04_clean_and_extract.py, same flags) so the
    cohorts stay comparable. Toggle with DO_REGISTER / DO_SKULLSTRIP.

Kaggle setup (first cell):
    !pip install -q SimpleITK nibabel nilearn
"""

import os, re, csv, glob, random, tempfile, time, json
from collections import defaultdict, Counter
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from PIL import Image
from scipy.ndimage import zoom

# ============================ CONFIG ============================
OUT       = "/kaggle/working/oasis"
IMG       = 224
SLICE_FRACS = [0.45, 0.50, 0.55]     # hippocampal band (same as ADNI)
PLANE_AXIS  = 1                      # coronal (RAS+ canonical)
VOL_SHAPE_3D = (128, 128, 128)
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED = 42
DO_N4 = True
DO_REGISTER   = True                 # affine-register each volume to MNI152
DO_SKULLSTRIP = True                 # mask out non-brain using the MNI152 brain mask
MNI_RES       = 2                    # MNI template resolution in mm (2 = fast, ample)
MATCH_WINDOW_DAYS = 730
CLASSES = ("CN", "MCI", "AD")
# ===============================================================

random.seed(SEED); np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)
OAS3_RE = re.compile(r"(OAS3\d{4,5})", re.IGNORECASE)
DAY_RE  = re.compile(r"[_\-]?d(\d{3,5})", re.IGNORECASE)


# ---- locate inputs under /kaggle/input ------------------------------------
def find_inputs():
    cdr, root = None, None
    for base in ("/kaggle/input", "."):
        for dp, dns, fns in os.walk(base):
            for f in fns:
                if f.lower().endswith(".csv") and "udsb4" in f.lower():
                    cdr = os.path.join(dp, f)
            if os.path.basename(dp).lower() == "oasis3_t1w":
                root = dp
    if not cdr:  # fallback: any csv with a cdr column later
        for base in ("/kaggle/input",):
            for dp, dns, fns in os.walk(base):
                for f in fns:
                    if f.lower().endswith(".csv") and "cdr" in f.lower():
                        cdr = os.path.join(dp, f)
    if not root or not cdr:
        raise SystemExit(f"Could not find oasis3_t1w/ and the CDR csv under /kaggle/input "
                         f"(root={root}, cdr={cdr})")
    return cdr, root


# ---- CDR labelling (nearest-assessment matching) --------------------------
def label_from_cdr(v):
    try: v = float(v)
    except (TypeError, ValueError): return None
    if v == 0: return "CN"
    if v == 0.5: return "MCI"
    if v >= 1: return "AD"
    return None


def read_cdr(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f)); fields = rows[0].keys() if rows else []
    fields = list(fields)
    def id_score(col):
        both = 0
        for r in rows[:500]:
            v = str(r.get(col, ""))
            if OAS3_RE.search(v) and DAY_RE.search(v):
                both += 1
        return both
    id_col = max(fields, key=id_score)
    low = {c.lower().strip(): c for c in fields}
    cdr_col = low.get("cdrtot") or low.get("cdrglob") or low.get("cdr")
    print(f"  CDR file: ID column '{id_col}', CDR column '{cdr_col}'")
    out = defaultdict(list)
    for r in rows:
        key = str(r.get(id_col, "")); m = OAS3_RE.search(key)
        if not m: continue
        dm = DAY_RE.search(key); day = int(dm.group(1)) if dm else 0
        try: cdr = float(r.get(cdr_col))
        except (TypeError, ValueError): continue
        out[m.group(1).upper()].append((day, cdr))
    return out


def index_images(root):
    # prefer .nii.gz; accept .nii; skip anything that isn't a real file (stray
    # directories named *.nii can appear as an unzip artifact and break the reader)
    out = defaultdict(list); seen = set()
    for pat in ("*T1w.nii.gz", "*t1w.nii.gz", "*T1w.nii", "*t1w.nii"):
        for p in glob.glob(os.path.join(root, "**", pat), recursive=True):
            if p in seen or not os.path.isfile(p):
                continue
            seen.add(p); blob = p.replace("\\", "/")
            m = OAS3_RE.search(blob)
            if not m: continue
            dm = DAY_RE.search(os.path.basename(blob)) or DAY_RE.search(blob)
            out[m.group(1).upper()].append((int(dm.group(1)) if dm else 0, p))
    return out


def label_subjects(cdr, images):
    rows, flags = [], []
    for sid in sorted(images):
        scans = sorted(images[sid]); assess = sorted(cdr.get(sid, []))
        if not assess:
            flags.append(f"{sid}: no CDR"); continue
        def gap(day): return min(abs(day - ad) for ad, _ in assess)
        scan_day, path = min(scans, key=lambda t: gap(t[0]))
        aday, val = min(assess, key=lambda t: abs(t[0] - scan_day))
        if abs(aday - scan_day) > MATCH_WINDOW_DAYS:
            flags.append(f"{sid}: CDR too far"); continue
        lab = label_from_cdr(val)
        if lab: rows.append((sid, lab, path))
    return rows, flags


# ---- imaging (same steps as 04_clean_and_extract.py) ----------------------
def n4_correct(img):
    img = sitk.Cast(img, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    shrink = sitk.Shrink(img, [2, 2, 2]); mshrink = sitk.Shrink(mask, [2, 2, 2])
    corr = sitk.N4BiasFieldCorrectionImageFilter()
    corr.SetMaximumNumberOfIterations([20, 20, 10])
    corr.Execute(shrink, mshrink)
    field = corr.GetLogBiasFieldAsImage(img)
    return sitk.Cast(img / sitk.Exp(field), sitk.sitkFloat32)


def sitk_to_canonical_array(img):
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "v.nii.gz"); sitk.WriteImage(img, tmp)
        vol = nib.as_closest_canonical(nib.load(tmp))
        return np.squeeze(np.asarray(vol.get_fdata(), dtype=np.float32))


# ---- MNI152 registration + skull-strip (nilearn ships the template & mask) ---
_MNI = {"tpl": None, "mask": None}
def _get_mni():
    """Load the MNI152 T1 template and its brain mask once, as SimpleITK images.
    nilearn bundles both in its package data, so this works offline on Kaggle."""
    if _MNI["tpl"] is None:
        from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
        td = tempfile.mkdtemp()
        tpl_p = os.path.join(td, "mni_t1.nii.gz"); msk_p = os.path.join(td, "mni_mask.nii.gz")
        nib.save(load_mni152_template(resolution=MNI_RES), tpl_p)
        nib.save(load_mni152_brain_mask(resolution=MNI_RES), msk_p)
        _MNI["tpl"]  = sitk.ReadImage(tpl_p, sitk.sitkFloat32)
        _MNI["mask"] = sitk.ReadImage(msk_p, sitk.sitkFloat32)  # same grid as tpl
    return _MNI["tpl"], _MNI["mask"]


def _reg_run(fixed, moving, tx_init):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.10, SEED)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(tx_init, inPlace=False)
    t = R.Execute(fixed, moving)
    return t, R.GetMetricValue()


def register_to_mni(moving):
    """Register to MNI152 with a MULTI-START strategy so scans whose orientation is far
    from the template still align: try (A) affine from a geometry init [the original
    single-stage method] and (C) a moments-initialised rigid stage then affine, and keep
    whichever fits best (lowest Mattes MI). Including (A) means this never does worse than
    before; (C) recovers the badly-oriented scans that were failing QC."""
    fixed, _ = _get_mni()
    f = sitk.Cast(fixed, sitk.sitkFloat32); m = sitk.Cast(moving, sitk.sitkFloat32)
    cands = []
    ta, va = _reg_run(f, m, sitk.CenteredTransformInitializer(
        f, m, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY))
    cands.append((va, sitk.Resample(m, f, ta, sitk.sitkLinear, 0.0, sitk.sitkFloat32)))
    tr, _ = _reg_run(f, m, sitk.CenteredTransformInitializer(
        f, m, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS))
    m_r = sitk.Resample(m, f, tr, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    tc, vc = _reg_run(f, m_r, sitk.CenteredTransformInitializer(
        f, m_r, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY))
    cands.append((vc, sitk.Resample(m_r, f, tc, sitk.sitkLinear, 0.0, sitk.sitkFloat32)))
    return min(cands, key=lambda x: x[0])[1]


def apply_brain_mask(vol_sitk):
    """Zero out non-brain voxels using the MNI152 brain mask (same grid as a
    registered volume). Removes skull/background."""
    _, mask = _get_mni()
    return sitk.Cast(vol_sitk * mask, sitk.sitkFloat32)


def normalise(vol):
    v = vol.astype(np.float32); m = v > np.percentile(v, 10)
    mu, sd = v[m].mean(), v[m].std() + 1e-6
    return (v - mu) / sd


def to_uint8(sl):
    lo, hi = np.percentile(sl, (1, 99)); s = np.clip(sl, lo, hi)
    return ((s - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)


def extract_slices(vol):
    n = vol.shape[PLANE_AXIS]; out = []
    for f in SLICE_FRACS:
        i = int(round(f * (n - 1)))
        sl = vol[:, i, :] if PLANE_AXIS == 1 else (vol[i, :, :] if PLANE_AXIS == 0 else vol[:, :, i])
        out.append(np.rot90(sl))
    return out


def subject_splits(items):
    bl = defaultdict(list)
    for s, l in items: bl[l].append(s)
    a = {}; tr, va, te = SPLIT_RATIOS
    for l, subs in bl.items():
        subs = sorted(set(subs)); random.Random(SEED).shuffle(subs)
        n = len(subs); ntr, nva = round(tr * n), round(va * n)
        for i, s in enumerate(subs):
            a[s] = "train" if i < ntr else "val" if i < ntr + nva else "test"
    return a


# ---- cumulative timing (survives stop/restart) ----------------------------
def _timing_path(cohort):
    return os.path.join(OUT, "..", f"clean_timing_{cohort}.json")

def load_timing(cohort):
    p = _timing_path(cohort)
    base = {"cohort": cohort, "cumulative_seconds": 0.0, "subjects_cleaned": 0, "runs": 0}
    if os.path.exists(p):
        try: base.update(json.load(open(p)))
        except Exception: pass
    return base

def save_timing(tinfo):
    json.dump(tinfo, open(_timing_path(tinfo["cohort"]), "w"), indent=2)

def finalize_timing(tinfo, script, run_t0, run_cleaned):
    save_timing(tinfo)
    cum_min = tinfo["cumulative_seconds"] / 60.0
    per = tinfo["cumulative_seconds"] / max(tinfo["subjects_cleaned"], 1)
    run_min = (time.time() - run_t0) / 60.0
    print(f"\nCLEANING TIME - cumulative across {tinfo['runs']} run(s): "
          f"{cum_min:.1f} min for {tinfo['subjects_cleaned']} subjects "
          f"({per:.1f} s/subject).  This run: {run_cleaned} subjects in {run_min:.1f} min.")
    # append a summary row to a shared run-times log
    log = os.path.join(OUT, "..", "run_times.csv"); new = not os.path.exists(log)
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["script", "cohort", "cumulative_min", "subjects", "runs", "timestamp"])
        w.writerow([script, tinfo["cohort"], round(cum_min, 2), tinfo["subjects_cleaned"],
                    tinfo["runs"], time.strftime("%Y-%m-%d %H:%M:%S")])


def main():
    cdr_csv, img_root = find_inputs()
    print("CDR:", cdr_csv, "\nscans:", img_root)
    cdr = read_cdr(cdr_csv); images = index_images(img_root)
    rows, flags = label_subjects(cdr, images)
    counts = Counter(l for _, l, _ in rows)
    print(f"\nlabelled {len(rows)} subjects  ->  "
          f"CN {counts.get('CN',0)}  MCI {counts.get('MCI',0)}  AD {counts.get('AD',0)}")
    if flags: print(f"  flagged/excluded: {len(flags)} (e.g. {flags[:3]})")

    # inventory
    with open(os.path.join(OUT, "..", "oasis_inventory.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject_id", "label", "t1w_path"]); w.writerows(rows)

    print(f"  preprocessing: N4={DO_N4}  register_MNI={DO_REGISTER}  skull_strip={DO_SKULLSTRIP}")
    tinfo = load_timing("oasis"); tinfo["runs"] += 1; save_timing(tinfo)
    run_t0, run_cleaned = time.time(), 0
    splits = subject_splits([(s, l) for s, l, _ in rows])
    os.makedirs(os.path.join(OUT, "volumes"), exist_ok=True)
    man = []
    for k, (sid, lab, src) in enumerate(rows, 1):
        sp = splits[sid]
        cls = os.path.join(OUT, "slices", sp, lab)
        expect = [os.path.join(cls, f"{sid}_s{i}.png") for i in range(len(SLICE_FRACS))]
        if all(os.path.exists(e) for e in expect):        # resumable: already sliced
            for e in expect: man.append([sid, lab, sp, e])
            print(f"  [{k}/{len(rows)}] {sid} skip (exists)")
            continue
        try:
            s0 = time.time()
            img = sitk.ReadImage(src)
            if DO_N4: img = n4_correct(img)
            if DO_REGISTER:   img = register_to_mni(img)
            if DO_SKULLSTRIP: img = apply_brain_mask(img)
            arr = sitk_to_canonical_array(img)
            os.makedirs(cls, exist_ok=True)
            for i, sl in enumerate(extract_slices(arr)):
                p = os.path.join(cls, f"{sid}_s{i}.png")
                Image.fromarray(to_uint8(sl)).convert("L").resize(
                    (IMG, IMG), Image.BILINEAR).save(p)
                man.append([sid, lab, sp, p])
            nv = normalise(arr); fac = [t / s for t, s in zip(VOL_SHAPE_3D, nv.shape)]
            nib.save(nib.Nifti1Image(zoom(nv, fac, order=1).astype(np.float32), np.eye(4)),
                     os.path.join(OUT, "volumes", f"{sid}.nii.gz"))
            dt = time.time() - s0
            tinfo["cumulative_seconds"] += dt; tinfo["subjects_cleaned"] += 1
            run_cleaned += 1; save_timing(tinfo)   # persist after each subject (crash-safe)
            print(f"  [{k}/{len(rows)}] {sid} ({lab}, {sp}) done  {dt:.1f}s")
        except Exception as e:
            print(f"  [{k}/{len(rows)}] {sid} FAILED: {e}")

    with open(os.path.join(OUT, "..", "oasis_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject_id", "label", "split", "path"]); w.writerows(man)
    finalize_timing(tinfo, "oasis_clean_kaggle.py", run_t0, run_cleaned)
    print(f"\nDONE. Wrote {len(man)} slices to {OUT}/slices")
    print("Split/class counts:")
    print(Counter((splits[s], l) for s, l, _ in rows))
    print("\nNow: zip /kaggle/working/oasis/slices and use it in the trainer "
          "(add 'oasis' to DATASETS).")


if __name__ == "__main__":
    main()
