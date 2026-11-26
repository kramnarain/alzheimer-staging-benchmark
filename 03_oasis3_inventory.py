"""
Build oasis_inventory.csv for the enlarged OASIS-3 v2 cohort DIRECTLY from the
205/205/205 shortlist (shortlist_labels_v2.csv), so the labels are exactly the ones
we selected (most-severe stage within 2 years) rather than re-derived by nearest-CDR.

Finds each subject's downloaded T1w NIfTI under the BIDS scans folder and writes the
same schema 04_clean_and_extract.py expects (subject_id, label, mpr1_file, has_image, ...).

Run:  python 03_oasis3_inventory.py
Then run 04_clean_and_extract.py (OASIS block) as usual.
"""
import os, re, csv, glob

# ---------------- CONFIG - EDIT THESE ----------------
SHORTLIST  = "data/OASIS-3/shortlist_labels_v2.csv"
SCANS_ROOT = "data/OASIS-3/oasis3_v2"        # BIDS scans root (sub-*/ses-*/anat/*T1w.nii*)
OUTPUT_CSV = "data/OASIS-3/oasis_inventory.csv"
# -----------------------------------------------------

FIELDS = ["subject_id","label","cdr","mmse","age","sex",
          "scan_day","cdr_day","match_gap_days","mpr1_file","has_image"]

def find_t1w(subj, day):
    pats = [
        os.path.join(SCANS_ROOT, f"sub-{subj}", f"ses-{day}", "anat", "*T1w.nii*"),
        os.path.join(SCANS_ROOT, f"sub-{subj}", "**", f"*{day}*T1w.nii*"),
        os.path.join(SCANS_ROOT, "**", f"*{subj}*{day}*T1w.nii*"),
    ]
    for p in pats:
        hits = sorted(glob.glob(p, recursive=True))
        hits = [h for h in hits if h.lower().endswith((".nii", ".nii.gz"))]
        if hits:
            return hits[0]
    return None

def main():
    with open(SHORTLIST, encoding="utf-8-sig", newline="") as f:
        rows_in = list(csv.DictReader(f))
    out, missing = [], []
    for r in rows_in:
        subj = r["subject_id"].strip()
        label = r["label"].strip()
        mrid = r.get("mr_id", "").strip()
        m = re.search(r"_MR_(d\d+)", mrid, re.I)
        day = m.group(1) if m else ""
        path = find_t1w(subj, day)
        if not path:
            missing.append(f"{subj} ({mrid})"); continue
        out.append({"subject_id": subj, "label": label, "cdr": "", "mmse": "",
                    "age": "", "sex": "", "scan_day": day.lstrip("d"),
                    "cdr_day": "", "match_gap_days": r.get("gap_days", ""),
                    "mpr1_file": path, "has_image": 1})
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(out)
    from collections import Counter
    c = Counter(r["label"] for r in out)
    print(f"Inventory written: {OUTPUT_CSV}")
    print(f"  subjects with a T1w found: {len(out)}  ->  CN {c.get('CN',0)}  MCI {c.get('MCI',0)}  AD {c.get('AD',0)}")
    if missing:
        print(f"  WARNING: {len(missing)} shortlisted subjects had no T1w under {SCANS_ROOT}:")
        for mdm in missing[:30]: print("   -", mdm)
        if len(missing) > 30: print(f"   ... and {len(missing)-30} more")

if __name__ == "__main__":
    main()
