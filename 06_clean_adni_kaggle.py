"""
ADNI-1 label + clean, on Kaggle (produces the 2D slices for training).

Mirrors 07_clean_oasis3_kaggle.py so the two cohorts get identical preprocessing:
N4 bias correction -> affine registration to MNI152 -> skull-strip with the MNI
brain mask -> three hippocampal-band 224x224 coronal slices -> subject-level
70/15/15 split. This is preprocessing v2 (registered + skull-stripped): the fixed
slice fractions now hit the same anatomy in every subject, and non-brain pixels are
removed.

Labels are built on Kaggle from the ADNI IDA metadata (researchGroup in the XMLs),
reusing the verified logic in 01_label_adni.py - so you do NOT need to upload a
label CSV, only the raw ADNI download (the one that contains the *.xml metadata and
the I<imageUID> DICOM folders).

INPUT (attach as a Kaggle dataset): the raw ADNI-1 download tree, containing both
the IDA *.xml metadata files and the I<uid> folders of .dcm images. The script finds
it automatically under /kaggle/input.

OUTPUT (to /kaggle/working):
    adni/slices/{train,val,test}/{CN,MCI,AD}/<subject>_s<k>.png
    adni/volumes/<subject>.nii.gz
    adni_inventory.csv , adni_manifest.csv

Then zip /kaggle/working/adni/slices and use it in the trainer (DATASETS=["adni"]).
Keep 01_label_adni.py beside this file.

Kaggle setup (first cell):
    !pip install -q SimpleITK nibabel nilearn
"""

import os, glob, csv, random, tempfile, importlib.util, time, json
from collections import defaultdict, Counter
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from PIL import Image
from scipy.ndimage import zoom

# ============================ CONFIG ============================
OUT       = "/kaggle/working/adni"
IMG       = 224
SLICE_FRACS = [0.45, 0.50, 0.55]     # hippocampal band (same as OASIS)
PLANE_AXIS  = 1                      # coronal (RAS+ canonical)
VOL_SHAPE_3D = (128, 128, 128)
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED = 42
DO_N4 = True
DO_REGISTER   = True                 # affine-register each volume to MNI152
DO_SKULLSTRIP = True                 # mask out non-brain using the MNI152 brain mask
MNI_RES       = 2
LABEL_FILE = "01_label_adni.py"      # reused for ADNI labelling
CLASSES = ("CN", "MCI", "AD")
# ===============================================================

random.seed(SEED); np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)


# ---- locate the raw ADNI download under /kaggle/input ---------------------
def find_adni_root():
    """A directory that (recursively) holds both IDA *.xml metadata and I<uid>
    DICOM folders. Return the shallowest such root."""
    for base in ("/kaggle/input", "."):
        if not os.path.isdir(base):
            continue
        for dp, dns, fns in os.walk(base):
            has_xml = any(f.lower().endswith(".xml") for f in fns)
            has_dcm_folder = any(d.startswith("I") and d[1:].isdigit() for d in dns)
            if has_xml or has_dcm_folder:
                # climb to a stable root: the dataset folder under /kaggle/input/<name>
                parts = dp.split(os.sep)
                if "input" in parts:
                    i = parts.index("input")
                    if len(parts) > i + 2:
                        return os.sep.join(parts[:i + 3])
                return dp
    return None


# ---- ADNI labelling (reuse 01_label_adni.py) ------------------------------
def _load(path, name):
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for c in (os.path.join(here, path), path, os.path.join(os.getcwd(), path)):
        if os.path.exists(c):
            spec = importlib.util.spec_from_file_location(name, c)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    raise FileNotFoundError(path)


def build_adni_inventory(root):
    """Return list of (subject_id, label, dcm_folder) using 01_label_adni's verified
    XML parsing and image indexing, without needing any local paths."""
    adl = _load(LABEL_FILE, "adni_label")
    adl.ADNI_ROOTS = [root]
    adl.MCI_ID_FILE = None                     # no whitelist on Kaggle
    keep = {adl.normalise_group(g) for g in adl.KEEP_GROUPS}

    by_subject = defaultdict(list)
    for p in glob.glob(os.path.join(root, "**", "*.xml"), recursive=True):
        rec = adl.parse_rich_xml(p)
        if rec and rec.get("subject_id"):
            by_subject[rec["subject_id"]].append(rec)
    image_index = adl.build_image_index(root)
    print(f"  ADNI: {len(by_subject)} labelled subjects, {len(image_index)} DICOM folders")

    rows, flags = [], []
    for subject, recs in sorted(by_subject.items()):
        cands = [r for r in recs if r["research_group"] in keep]
        if not cands:
            flags.append(subject); continue
        cands.sort(key=lambda r: (
            0 if ("I" + (r["image_uid"] or "")) in image_index else 1,
            adl.variant_rank(r), r["image_uid"] or ""))
        chosen = cands[0]
        folder = image_index.get("I" + (chosen["image_uid"] or ""))
        if folder and adl.count_dcm(folder) > 0:
            rows.append((subject, chosen["research_group"], folder))
        else:
            flags.append(subject)
    if flags:
        print(f"  ADNI: {len(flags)} subject(s) skipped (no usable DICOM): {flags[:3]}...")
    return rows


# ---- imaging (identical to oasis_clean_kaggle.py) -------------------------
def load_dicom(folder):
    r = sitk.ImageSeriesReader()
    ids = r.GetGDCMSeriesIDs(folder)
    files = r.GetGDCMSeriesFileNames(folder, ids[0]) if ids else sorted(glob.glob(os.path.join(folder, "*.dcm")))
    r.SetFileNames(files)
    return r.Execute()


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


_MNI = {"tpl": None, "mask": None}
def _get_mni():
    if _MNI["tpl"] is None:
        from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
        td = tempfile.mkdtemp()
        tpl_p = os.path.join(td, "mni_t1.nii.gz"); msk_p = os.path.join(td, "mni_mask.nii.gz")
        nib.save(load_mni152_template(resolution=MNI_RES), tpl_p)
        nib.save(load_mni152_brain_mask(resolution=MNI_RES), msk_p)
        _MNI["tpl"]  = sitk.ReadImage(tpl_p, sitk.sitkFloat32)
        _MNI["mask"] = sitk.ReadImage(msk_p, sitk.sitkFloat32)
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
    """Multi-start MNI152 registration (see oasis_clean_kaggle.py): try affine-from-geometry
    and moments-rigid-then-affine, keep the best-fitting (lowest Mattes MI). Recovers
    badly-oriented scans without ever doing worse than the single-stage method."""
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
    log = os.path.join(OUT, "..", "run_times.csv"); new = not os.path.exists(log)
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["script", "cohort", "cumulative_min", "subjects", "runs", "timestamp"])
        w.writerow([script, tinfo["cohort"], round(cum_min, 2), tinfo["subjects_cleaned"],
                    tinfo["runs"], time.strftime("%Y-%m-%d %H:%M:%S")])


def main():
    root = find_adni_root()
    if not root:
        raise SystemExit("Could not find the raw ADNI download under /kaggle/input "
                         "(need the folder with *.xml metadata and I<uid> DICOM folders).")
    print("ADNI root:", root)
    rows = build_adni_inventory(root)
    counts = Counter(l for _, l, _ in rows)
    print(f"\nlabelled {len(rows)} subjects  ->  "
          f"CN {counts.get('CN',0)}  MCI {counts.get('MCI',0)}  AD {counts.get('AD',0)}")
    if not rows:
        raise SystemExit("No labelled ADNI subjects found - check the uploaded dataset.")

    with open(os.path.join(OUT, "..", "adni_inventory.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject_id", "label", "dcm_folder"]); w.writerows(rows)

    print(f"  preprocessing: N4={DO_N4}  register_MNI={DO_REGISTER}  skull_strip={DO_SKULLSTRIP}")
    tinfo = load_timing("adni"); tinfo["runs"] += 1; save_timing(tinfo)
    run_t0, run_cleaned = time.time(), 0
    splits = subject_splits([(s, l) for s, l, _ in rows])
    os.makedirs(os.path.join(OUT, "volumes"), exist_ok=True)
    man = []
    for k, (sid, lab, src) in enumerate(rows, 1):
        sp = splits[sid]
        cls = os.path.join(OUT, "slices", sp, lab)
        expect = [os.path.join(cls, f"{sid}_s{i}.png") for i in range(len(SLICE_FRACS))]
        if all(os.path.exists(e) for e in expect):        # resumable
            for e in expect: man.append([sid, lab, sp, e])
            print(f"  [{k}/{len(rows)}] {sid} skip (exists)")
            continue
        try:
            s0 = time.time()
            img = load_dicom(src)
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

    with open(os.path.join(OUT, "..", "adni_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject_id", "label", "split", "path"]); w.writerows(man)
    finalize_timing(tinfo, "adni_clean_kaggle.py", run_t0, run_cleaned)
    print(f"\nDONE. Wrote {len(man)} slices to {OUT}/slices")
    print("Split/class counts:", Counter((splits[s], l) for s, l, _ in rows))
    print("\nNow: zip /kaggle/working/adni/slices and use it in the trainer (DATASETS=['adni']).")


if __name__ == "__main__":
    main()
