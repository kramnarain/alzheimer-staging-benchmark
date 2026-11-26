"""
Local preprocessing and feature extraction - the core of the pipeline.

ADNI and OASIS: N4 bias-field correction, intensity normalisation, three
hippocampal-band coronal slices per subject at 224x224, plus a resampled 3D volume.
Kaggle images are just standardised to the same size. Splits are stratified and
subject-level (70/15/15). Features are scikit-image radiomics plus ResNet-50
embeddings, concatenated into features_all.parquet.

Built to run in the background: capped to a couple of threads, below-normal Windows
priority, and a per-subject "done" marker so it picks up where it left off after a
sleep, crash or manual stop. Feature extraction is incremental - only slices missing
from the parquet get the (slow) embedding pass.
"""

import os

# ---- thread caps MUST be set before importing numpy/torch/SimpleITK ----------
CLEAN_THREADS = 2                       # CPU threads to use (leave cores for your VM)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(CLEAN_THREADS)

import csv
import glob
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from PIL import Image
from scipy.ndimage import zoom
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, canny
from skimage.filters import sobel, threshold_otsu

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(CLEAN_THREADS)

# lower process priority on Windows so the VM / browser stay responsive
try:
    import ctypes
    BELOW_NORMAL = 0x00004000
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
    print("Process priority set to below-normal.")
except Exception:
    pass

# =============================================================================
# CONFIG
# =============================================================================
# ADNI reuse. A previously cleaned ADNI set may not be registered/skull-stripped, so with the
# v2 preprocessing below it can no longer be reused as-is (it would be inconsistent
# with the registered OASIS/AIBL). To re-clean ADNI from raw DICOM through the SAME
# registered pipeline, set REUSE_ADNI_CLEANED = None and add an ADNI inventory/source
# so clean_volumetric("adni", ...) runs. Leave the path only if you deliberately want
# the old unregistered ADNI (not recommended once DO_REGISTER=True).
REUSE_ADNI_CLEANED = None    # None => clean ADNI through the registered pipeline here

OASIS_CSV = "data/OASIS-3/oasis_inventory.csv"      # from 03_oasis3_inventory.py
AIBL_CSV  = "data/AIBL/aibl_inventory.csv"          # optional, unused by default

OUT = "data/processed_clean"

DATASETS = ("adni", "oasis")     # 'oasis' holds the OASIS-3 data

IMG_SIZE     = 224
SLICE_FRACS  = [0.45, 0.50, 0.55]     # hippocampal band
PLANE_AXIS   = 1                      # coronal (RAS+ canonical)
VOL_SHAPE_3D = (128, 128, 128)
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED         = 42
DO_N4        = True
DO_REGISTER   = True                 # affine-register each volume to MNI152
DO_SKULLSTRIP = True                 # mask out non-brain with the MNI152 brain mask
MNI_RES       = 2                    # MNI template resolution in mm
PREPROC_VER   = "mni_v2"             # bumped so old (unregistered) done-markers are ignored
# =============================================================================

random.seed(SEED); np.random.seed(SEED)
LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}     # 3-class ordinal scheme
CLASSES = ("CN", "MCI", "AD")
CKPT = os.path.join(OUT, "_clean_ckpt")
os.makedirs(CKPT, exist_ok=True)


# ------------------------------------------------------------------ imaging
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
    """Write to a temp NIfTI and reload canonical so ADNI/OASIS share orientation."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "v.nii.gz")
        sitk.WriteImage(img, tmp)
        vol = nib.as_closest_canonical(nib.load(tmp))
        return np.squeeze(np.asarray(vol.get_fdata(), dtype=np.float32))


# ---- MNI152 registration + skull-strip (nilearn ships the template & mask) ---
_MNI = {"tpl": None, "mask": None}
def _get_mni():
    """Load the MNI152 T1 template and brain mask once, as SimpleITK images."""
    if _MNI["tpl"] is None:
        import tempfile
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
    """Multi-start MNI152 registration: try affine-from-geometry and
    moments-rigid-then-affine, keep the best-fitting (lowest Mattes MI). Recovers
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
    """Zero out non-brain voxels using the MNI152 brain mask (same grid as a
    registered volume)."""
    _, mask = _get_mni()
    return sitk.Cast(vol_sitk * mask, sitk.sitkFloat32)


def load_adni(dcm_folder):
    r = sitk.ImageSeriesReader()
    ids = r.GetGDCMSeriesIDs(dcm_folder)
    files = r.GetGDCMSeriesFileNames(dcm_folder, ids[0]) if ids else sorted(glob.glob(os.path.join(dcm_folder, "*.dcm")))
    r.SetFileNames(files)
    return r.Execute()


def load_volume(ds, src):
    """Generic loader. ADNI = DICOM series folder. OASIS-3 = a NIfTI file path.
    AIBL = either a DICOM series folder (has .dcm) or a folder holding one NIfTI."""
    if ds == "adni":
        return load_adni(src)
    if os.path.isdir(src):
        if glob.glob(os.path.join(src, "*.dcm")):
            return load_adni(src)                      # DICOM series in a folder
        nii = glob.glob(os.path.join(src, "*.nii")) + glob.glob(os.path.join(src, "*.nii.gz"))
        if nii:
            return sitk.ReadImage(sorted(nii)[0])
        raise FileNotFoundError(f"no .dcm or .nii(.gz) in folder {src}")
    return sitk.ReadImage(src)                          # a NIfTI/Analyze file path


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


def done_marker(ds, sid):
    # PREPROC_VER in the name means switching to registration invalidates old markers
    return os.path.join(CKPT, f"{ds}__{sid}__{PREPROC_VER}.done")


# ------------------------------------------------------------------ cleaning
def clean_volumetric(ds, rows):
    """rows: list of (subject_id, label, source_path). ADNI=dcm folder, OASIS=mpr1 file."""
    splits = subject_splits([(s, l) for s, l, _ in rows])
    os.makedirs(os.path.join(OUT, ds, "volumes"), exist_ok=True)
    total = len(rows)
    for k, (sid, lab, src) in enumerate(rows, 1):
        if os.path.exists(done_marker(ds, sid)):
            continue
        try:
            img = load_volume(ds, src)
            if DO_N4:
                img = n4_correct(img)
            if DO_REGISTER:
                img = register_to_mni(img)
            if DO_SKULLSTRIP:
                img = apply_brain_mask(img)
            arr = sitk_to_canonical_array(img)
            sp = splits[sid]
            cls = os.path.join(OUT, ds, "slices", sp, lab); os.makedirs(cls, exist_ok=True)
            for i, sl in enumerate(extract_slices(arr)):
                Image.fromarray(to_uint8(sl)).convert("L").resize(
                    (IMG_SIZE, IMG_SIZE), Image.BILINEAR).save(os.path.join(cls, f"{sid}_s{i}.png"))
            nv = normalise(arr); fac = [t / s for t, s in zip(VOL_SHAPE_3D, nv.shape)]
            nib.save(nib.Nifti1Image(zoom(nv, fac, order=1).astype(np.float32), np.eye(4)),
                     os.path.join(OUT, ds, "volumes", f"{sid}.nii.gz"))
            open(done_marker(ds, sid), "w").write("ok")
            print(f"  [{ds}] {k}/{total} {sid} done")
        except Exception as e:
            print(f"  [{ds}] {k}/{total} {sid} FAILED: {e}")


def clean_kaggle(rows):
    """rows: list of (image_id, label, path)."""
    splits = subject_splits([(iid, lab) for iid, lab, _ in rows])
    done = os.path.join(CKPT, "kaggle.done")
    if os.path.exists(done):
        print("  kaggle already standardised."); return
    for iid, lab, src in rows:
        if not os.path.exists(src):
            continue
        sp = splits[iid]; cls = os.path.join(OUT, "kaggle", "slices", sp, lab); os.makedirs(cls, exist_ok=True)
        safe = iid.replace("/", "__").replace("\\", "__")
        a = np.asarray(Image.open(src).convert("L"), dtype=np.float32)
        Image.fromarray(to_uint8(a)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR).save(os.path.join(cls, safe))
    open(done, "w").write("ok")
    print("  kaggle done.")


def build_manifest():
    rows = []
    for ds in DATASETS:
        base = os.path.join(OUT, ds, "slices")
        slice_paths = []
        for ext in ("png", "jpg", "jpeg"):
            slice_paths += glob.glob(os.path.join(base, "*", "*", f"*.{ext}"))
        for p in slice_paths:
            parts = p.split(os.sep)
            split, label, fname = parts[-3], parts[-2], parts[-1]
            sid = fname.rsplit("_s", 1)[0]
            rows.append([ds, sid, label, split, "2d", 0, p])
        for p in glob.glob(os.path.join(OUT, ds, "volumes", "*.nii.gz")):
            sid = os.path.basename(p)[:-7]
            rows.append([ds, sid, "", "", "3d", -1, p])
    man = pd.DataFrame(rows, columns=["dataset", "subject_id", "label", "split", "kind", "slice_index", "path"])
    man.to_csv(os.path.join(OUT, "manifest.csv"), index=False)
    return man


# ------------------------------------------------------------------ features
def radiomic(g):
    f = {}; img = g.astype(np.float32)
    try: thr = threshold_otsu(g)
    except Exception: thr = 0
    mask = g > max(thr, 1); vals = img[mask] if mask.any() else img.ravel()
    f["rad_mean"] = float(vals.mean()); f["rad_std"] = float(vals.std()); f["rad_var"] = float(vals.var())
    f["rad_min"] = float(vals.min()); f["rad_max"] = float(vals.max()); f["rad_range"] = float(vals.max() - vals.min())
    f["rad_median"] = float(np.median(vals))
    for p in (10, 25, 75, 90): f[f"rad_p{p}"] = float(np.percentile(vals, p))
    f["rad_iqr"] = f["rad_p75"] - f["rad_p25"]
    f["rad_skew"] = float(skew(vals)) if vals.std() > 0 else 0.0
    f["rad_kurt"] = float(kurtosis(vals)) if vals.std() > 0 else 0.0
    f["rad_mad"] = float(np.mean(np.abs(vals - vals.mean()))); f["rad_rms"] = float(np.sqrt(np.mean(vals ** 2)))
    f["rad_energy"] = float(np.sum(vals ** 2) / 1e6)
    h = np.histogram(vals, bins=32, range=(0, 255))[0].astype(np.float32); pp = h / (h.sum() + 1e-9)
    f["rad_entropy"] = float(-np.sum(pp * np.log2(pp + 1e-9))); f["rad_uniformity"] = float(np.sum(pp ** 2))
    q = (g.astype(np.uint16) * 31 // 255).astype(np.uint8); ang = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    for d in (1, 2, 3):
        gl = graycomatrix(q, [d], ang, levels=32, symmetric=True, normed=True)
        for pr in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
            f[f"rad_glcm_{pr}_d{d}"] = float(np.mean(graycoprops(gl, pr)))
    lbp = local_binary_pattern(g, 8, 1, method="uniform"); lh = np.histogram(lbp, bins=10, range=(0, 10))[0].astype(np.float32); lh /= (lh.sum() + 1e-9)
    for i, v in enumerate(lh): f[f"rad_lbp_{i}"] = float(v)
    gg = sobel(img / 255.0); f["rad_grad_mean"] = float(gg.mean()); f["rad_grad_std"] = float(gg.std())
    f["rad_edge_frac"] = float(canny(img / 255.0, sigma=2.0).mean()); f["rad_fg_frac"] = float(mask.mean())
    ys, xs = np.where(mask)
    if len(xs) > 0:
        h2 = ys.max() - ys.min() + 1; w2 = xs.max() - xs.min() + 1
        f["rad_bbox_ratio"] = float(h2 / (w2 + 1e-9)); f["rad_extent"] = float(mask.sum() / (h2 * w2 + 1e-9))
    else:
        f["rad_bbox_ratio"] = 0.0; f["rad_extent"] = 0.0
    return f


def extract_features(man):
    import torch
    torch.set_num_threads(CLEAN_THREADS)
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); model.fc = torch.nn.Identity(); model.eval()
    tf = transforms.Compose([transforms.Grayscale(3), transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def embed(paths, bs=32):
        outs = []
        with torch.no_grad():
            for i in range(0, len(paths), bs):
                xb = torch.stack([tf(Image.open(p).convert("L")) for p in paths[i:i + bs]])
                outs.append(model(xb).cpu().numpy())
                if (i // bs) % 25 == 0:
                    print(f"      embedded {i + len(paths[i:i+bs])}/{len(paths)}")
        return np.vstack(outs).astype(np.float32)

    fdir = os.path.join(OUT, "features"); os.makedirs(fdir, exist_ok=True)
    two_d = man[man["kind"] == "2d"]
    parts = []
    for ds in DATASETS:
        outp = os.path.join(fdir, f"features_{ds}.parquet")
        d = two_d[two_d.dataset == ds]
        if d.empty:
            continue

        # INCREMENTAL: only embed slices that aren't already in the feature file.
        # (After a relabel/resplit the existing rows keep their features and just
        #  carry updated label/split/path metadata, so only genuinely new subjects
        #  - e.g. the added MCI cohort - need the expensive ResNet-50 pass.)
        existing = None
        if os.path.exists(outp):
            existing = pd.read_parquet(outp)
            have = set(existing["path"].map(os.path.basename))
            todo = d[~d.path.map(os.path.basename).isin(have)]
            if todo.empty:
                print(f"  features_{ds} up to date ({len(existing)} slices).")
                parts.append(existing)
                continue
            print(f"  features_{ds}: {len(existing)} existing, {len(todo)} NEW slices to extract")
        else:
            todo = d
            print(f"  extracting features: {ds} ({len(todo)} slices)")

        rad = pd.DataFrame([radiomic(np.asarray(Image.open(p).convert("L"), dtype=np.uint8)) for p in todo.path])
        emb = embed(todo.path.tolist()); embdf = pd.DataFrame(emb, columns=[f"emb_{k}" for k in range(emb.shape[1])])
        meta = todo[["dataset", "subject_id", "label", "split", "path"]].reset_index(drop=True)
        out = pd.concat([meta, rad.reset_index(drop=True), embdf], axis=1)
        if existing is not None:
            out = pd.concat([existing, out], ignore_index=True)
        fc = [c for c in out.columns if c.startswith(("rad_", "emb_"))]
        out[fc] = out[fc].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out.to_parquet(outp, index=False); parts.append(out)
    if parts:
        allf = pd.concat(parts, ignore_index=True)
        allf.to_parquet(os.path.join(fdir, "features_all.parquet"), index=False)
        print(f"\nfeatures_all.parquet: {allf.shape}")


def reuse_adni():
    """Copy a previously cleaned ADNI set (slices, volumes, feature parquet) into OUT so
    the manifest and feature steps include ADNI without re-cleaning it. Idempotent:
    skips anything already present."""
    if not REUSE_ADNI_CLEANED:
        print("REUSE_ADNI_CLEANED is None - expecting an adni/ folder already in OUT.")
        return
    import shutil
    src_adni = os.path.join(REUSE_ADNI_CLEANED, "adni")
    dst_adni = os.path.join(OUT, "adni")
    if os.path.isdir(src_adni) and not os.path.isdir(dst_adni):
        print(f"  copying reused ADNI slices/volumes: {src_adni} -> {dst_adni}")
        shutil.copytree(src_adni, dst_adni)
    else:
        print(f"  ADNI reuse: {'already in OUT' if os.path.isdir(dst_adni) else 'source not found: '+src_adni}")
    # reuse the ADNI feature parquet so the ResNet-50 pass is skipped for ADNI
    fdir = os.path.join(OUT, "features"); os.makedirs(fdir, exist_ok=True)
    src_feat = os.path.join(REUSE_ADNI_CLEANED, "features", "features_adni.parquet")
    dst_feat = os.path.join(fdir, "features_adni.parquet")
    if os.path.exists(src_feat) and not os.path.exists(dst_feat):
        shutil.copy2(src_feat, dst_feat)
        print("  copied features_adni.parquet (ADNI embeddings reused)")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Output -> {OUT}   (threads={CLEAN_THREADS}, N4={DO_N4}). Resumable.\n")

    # OASIS-3 v2 run: ADNI added separately (step 2), no AIBL (no permission),
    # and the radiomic/ResNet-50 parquet is skipped because the J3 deep models read
    # the slices directly. Flip these to re-enable.
    DO_ADNI     = False
    DO_AIBL     = False
    DO_FEATURES = False

    if DO_ADNI:
        print("--- ADNI (reused from a previous clean, not re-cleaned) ---")
        reuse_adni()

    print("\n--- OASIS-3 ---")
    oa = pd.read_csv(OASIS_CSV); oa = oa[oa["has_image"] == 1]
    clean_volumetric("oasis", [(str(r.subject_id), r.label, r.mpr1_file) for r in oa.itertuples()])

    if DO_AIBL:
        print("\n--- AIBL ---")
        ai = pd.read_csv(AIBL_CSV); ai = ai[ai["n_files"] > 0]
        clean_volumetric("aibl", [(str(r.subject_id), r.label, r.image_path) for r in ai.itertuples()])

    print("\n--- Manifest ---")
    man = build_manifest()
    print("manifest rows:", len(man))
    print(man[man.kind == "2d"].groupby(["dataset", "split", "label"]).size().to_string())

    if DO_FEATURES:
        print("\n--- Feature extraction ---")
        extract_features(man)
    print(f"\nALL DONE. Point the training scripts at {OUT} and run them.")


if __name__ == "__main__":
    main()
