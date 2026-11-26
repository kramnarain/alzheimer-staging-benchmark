"""
ADNI-1 inventory builder.

One row per subject: its CN / MCI / AD label and the path to the MP-RAGE DICOM
folder. Labels come straight from the IDA metadata XMLs (researchGroup); each is
matched to its image folder on disk, preferring a plain MP-RAGE and falling back
to a repeat only when that is all a subject has. Scans both the original AD/CN
download and the later MCI download. Writes adni_inventory.csv.

Only folders that actually contain .dcm files are indexed - the extracted metadata
archives mirror the folder skeleton but hold no images, and would otherwise shadow
the real scans.
"""

import os
import re
import csv
import glob
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# CONFIG  -  3-CLASS VERSION (CN / MCI / AD)
# ---------------------------------------------------------------------------
# Top-level download roots. Each is searched RECURSIVELY for both the IDA
# metadata XMLs and the I<imageUID> DICOM folders, so the internal layout of
# each download doesn't matter. Add the MCI download as a second root.
ADNI_ROOTS = [
    "data/ADNI",          # set to your ADNI download root (searched recursively)
]

# Where to write the output CSV:
OUTPUT_CSV = "data/ADNI/adni_inventory.csv"

# Groups to keep (3-class task). EMCI/LMCI are folded into MCI if they appear.
KEEP_GROUPS = {"AD", "CN", "MCI", "EMCI", "LMCI"}

# Optional safety net: if this file exists, MCI subjects are restricted to the
# IDs listed in it (the site-stratified sample of 100). Set to None to disable.
MCI_ID_FILE = None  # if set, restrict MCI to the IDs listed in that file
# ---------------------------------------------------------------------------


def normalise_group(g):
    """Map the raw researchGroup to one of CN / MCI / AD."""
    g = (g or "").strip().upper()
    if g in ("MCI", "EMCI", "LMCI"):
        return "MCI"
    return g


def load_mci_whitelist():
    """{subject_id: {sex, age}} for the approved MCI subset, or None."""
    if not MCI_ID_FILE or not os.path.exists(MCI_ID_FILE):
        return None
    info = {}
    with open(MCI_ID_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = (row.get("Subject ID") or row.get("subject_id") or "").strip()
            if sid:
                info[sid] = {"sex": (row.get("Sex") or "").strip(),
                             "age": (row.get("Age") or "").strip()}
    return info or None


SUBJECT_RE = re.compile(r"^\d{3}_S_\d{4}$")


def index_subject_images(roots):
    """Walk the download tree and map subject_id -> [(series_folder_name, I*path)].

    Used as a fallback when a cohort was downloaded WITHOUT the IDA metadata
    XMLs (the MCI collection). The on-disk folder name carries the series
    description (e.g. 'MP-RAGE' or 'MP-RAGE_REPEAT'), which is enough to rank
    the scans exactly the same way the XML path does.
    """
    out = defaultdict(list)
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _ in os.walk(root):
            if not SUBJECT_RE.match(os.path.basename(dirpath)):
                continue
            sid = os.path.basename(dirpath)
            for series in os.listdir(dirpath):
                spath = os.path.join(dirpath, series)
                if not os.path.isdir(spath):
                    continue
                for sub, _dn, fn in os.walk(spath):
                    b = os.path.basename(sub)
                    # same trap as build_image_index: skip empty metadata stubs
                    if b.startswith("I") and b[1:].isdigit() and \
                            any(f.lower().endswith(".dcm") for f in fn):
                        out[sid].append((series, sub))
    return out


def folder_rank(series_name):
    """Same preference order as variant_rank, but from the on-disk folder name."""
    n = series_name.upper()
    if "REPEAT" in n:
        return 3
    if "24_FOV" in n or "24 FOV" in n:
        return 2
    return 1


def local(tag):
    """Strip any XML namespace, returning just the local tag name."""
    return tag.rsplit("}", 1)[-1]


def find_text(root, name):
    """First element anywhere under root whose local tag == name; else None."""
    for el in root.iter():
        if local(el.tag) == name:
            return (el.text or "").strip()
    return None


def find_protocol(root, term):
    """Value of <protocol term="..."> under imagingProtocol; else None."""
    for el in root.iter():
        if local(el.tag) == "protocol" and el.attrib.get("term") == term:
            return (el.text or "").strip()
    return None


def find_assessment(root, attribute):
    """Value of <assessmentScore attribute="..."> (e.g. CDGLOBAL, MMSCORE)."""
    for el in root.iter():
        if local(el.tag) == "assessmentScore" and el.attrib.get("attribute") == attribute:
            return (el.text or "").strip()
    return None


def parse_rich_xml(path):
    """Return a dict of fields for a rich IDA metadata XML, or None if it
    isn't one (i.e. has no researchGroup)."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None

    group = find_text(root, "researchGroup")
    if not group:
        return None  # this is one of the tiny pointer XMLs, skip it

    return {
        "subject_id":     find_text(root, "subjectIdentifier"),
        "research_group": normalise_group(group),
        "image_uid":      find_text(root, "imageUID"),
        "series_uid":     find_text(root, "seriesIdentifier"),
        "description":    (find_text(root, "description") or "").strip(),
        "field_strength": find_protocol(root, "Field Strength"),
        "cdr_global":     find_assessment(root, "CDGLOBAL"),
        "mmse":           find_assessment(root, "MMSCORE"),
        "sex":            find_text(root, "subjectSex"),
        "age":            find_text(root, "subjectAge"),
        "source_xml":     os.path.basename(path),
    }


def is_repeat(rec):
    blob = (rec["description"] + " " + rec["source_xml"]).upper()
    return "REPEAT" in blob


def variant_rank(rec):
    """Lower = more preferred.

        1  plain MP-RAGE
        2  MP-RAGE_24_FOV
        3  REPEAT  - fallback ONLY. A handful of ADNI subjects (e.g. 126_S_0708,
           126_S_0709, 126_S_1187) have no plain acquisition at all; a repeat is
           a valid same-session T1 re-acquisition, so using it keeps the subject
           rather than silently dropping it.
    """
    blob = (rec["description"] + " " + rec["source_xml"]).upper()
    if is_repeat(rec):
        return 3
    if "24_FOV" in blob or "24 FOV" in blob:
        return 2
    return 1  # plain MP-RAGE


def build_image_index(image_root):
    """Map every image folder named I<digits> to its full path.

    IMPORTANT: only folders that actually CONTAIN .dcm files are indexed. The
    extracted IDA *metadata* archives mirror the same subject/series/I<uid>
    directory skeleton but with no images in it. Indexing on folder name alone
    let those empty stubs overwrite the real image paths (dict key collision),
    so every subject resolved to a folder holding 0 DICOMs. If the same I<uid>
    appears twice, the copy with the most DICOMs wins.
    """
    best = {}
    for dirpath, _dirnames, filenames in os.walk(image_root):
        base = os.path.basename(dirpath)
        if not (base.startswith("I") and base[1:].isdigit()):
            continue
        n = sum(1 for f in filenames if f.lower().endswith(".dcm"))
        if n == 0:
            continue                                   # empty metadata stub
        if base not in best or n > best[base][1]:
            best[base] = (dirpath, n)
    return {k: v[0] for k, v in best.items()}


def count_dcm(folder):
    return sum(1 for f in os.listdir(folder) if f.lower().endswith(".dcm"))


def main():
    mci_whitelist = load_mci_whitelist()
    if mci_whitelist:
        print(f"MCI whitelist active: {len(mci_whitelist)} approved subject IDs\n")

    by_subject = defaultdict(list)
    image_index = {}
    for root in ADNI_ROOTS:
        if not os.path.isdir(root):
            print(f"!! root not found, skipping: {root}")
            continue
        print("Scanning root:", root)
        xml_files = glob.glob(os.path.join(root, "**", "*.xml"), recursive=True)
        n_before = len(by_subject)
        for p in xml_files:
            rec = parse_rich_xml(p)
            if rec and rec["subject_id"]:
                by_subject[rec["subject_id"]].append(rec)
        idx = build_image_index(root)
        image_index.update(idx)
        print(f"  {len(xml_files)} XMLs, {len(idx)} image folders, "
              f"+{len(by_subject) - n_before} new subjects")

    print(f"\n{len(by_subject)} subjects have usable (labelled) metadata")
    print(f"{len(image_index)} image (I*) folders indexed on disk")

    rows = []
    flags = []
    for subject, recs in sorted(by_subject.items()):
        # keep all CN/MCI/AD scans; repeats are ranked last and used only as a
        # fallback (see variant_rank), so no subject is lost for lacking a plain scan
        candidates = [r for r in recs if r["research_group"] in
                      {normalise_group(g) for g in KEEP_GROUPS}]
        # honour the MCI subset if this subject is MCI
        if candidates and candidates[0]["research_group"] == "MCI" and mci_whitelist:
            if subject not in mci_whitelist:
                continue
        if not candidates:
            flags.append(f"{subject}: no CN/MCI/AD scan in metadata "
                         f"(groups seen: {sorted({r['research_group'] for r in recs})})")
            continue

        # Prefer (a) scans whose DICOM folder is actually on disk - you deleted the
        # redundant repeat folders - then (b) plain over 24_FOV over repeat, then
        # (c) image_uid for determinism.
        candidates.sort(key=lambda r: (
            0 if ("I" + (r["image_uid"] or "")) in image_index else 1,
            variant_rank(r),
            r["image_uid"] or "",
        ))
        chosen = candidates[0]

        # locate the DICOM folder on disk
        folder = image_index.get("I" + (chosen["image_uid"] or ""))
        if folder is None:
            flags.append(f"{subject}: chosen image I{chosen['image_uid']} "
                         f"({chosen['description']}) has no folder on disk")
            n_dcm, folder_str = 0, ""
        else:
            n_dcm = count_dcm(folder)
            folder_str = folder
            if n_dcm == 0:
                flags.append(f"{subject}: folder {folder} contains 0 .dcm files")

        label = chosen["research_group"]          # CN / MCI / AD
        rows.append({
            "subject_id":     subject,
            "label":          label,
            "research_group": chosen["research_group"],
            "description":    chosen["description"],
            "variant":        {1: "MP-RAGE", 2: "MP-RAGE_24_FOV",
                               3: "MP-RAGE_REPEAT (fallback)"}[variant_rank(chosen)],
            "series_uid":     chosen["series_uid"],
            "image_uid":      chosen["image_uid"],
            "field_strength": chosen["field_strength"],
            "cdr_global":     chosen["cdr_global"],
            "mmse":           chosen["mmse"],
            "sex":            chosen["sex"],
            "age":            chosen["age"],
            "n_dcm":          n_dcm,
            "dcm_folder":     folder_str,
        })

    # ---- fallback: MCI subjects downloaded WITHOUT metadata XMLs -------------
    if mci_whitelist:
        already = {r["subject_id"] for r in rows}
        disk = index_subject_images(ADNI_ROOTS)
        added = 0
        for sid, demo in sorted(mci_whitelist.items()):
            if sid in already:
                continue
            series = disk.get(sid)
            if not series:
                flags.append(f"{sid}: MCI in whitelist but no image folder found on disk")
                continue
            series = sorted(series, key=lambda t: (folder_rank(t[0]), t[1]))
            sname, ipath = series[0]
            n_dcm = count_dcm(ipath)
            if n_dcm == 0:
                flags.append(f"{sid}: MCI folder {ipath} contains 0 .dcm files")
            rows.append({
                "subject_id": sid, "label": "MCI", "research_group": "MCI",
                "description": sname,
                "variant": {1: "MP-RAGE", 2: "MP-RAGE_24_FOV",
                            3: "MP-RAGE_REPEAT (fallback)"}[folder_rank(sname)],
                "series_uid": "", "image_uid": os.path.basename(ipath).lstrip("I"),
                "field_strength": "1.5", "cdr_global": "", "mmse": "",
                "sex": demo.get("sex", ""), "age": demo.get("age", ""),
                "n_dcm": n_dcm, "dcm_folder": ipath,
            })
            added += 1
        if added:
            print(f"\nMCI fallback (no XML metadata): recovered {added} subjects from disk")
        rows.sort(key=lambda r: r["subject_id"])

    # write CSV
    fieldnames = ["subject_id", "label", "research_group", "description", "variant",
                  "series_uid", "image_uid", "field_strength", "cdr_global",
                  "mmse", "sex", "age", "n_dcm", "dcm_folder"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # summary
    print("\n" + "=" * 60)
    print(f"Inventory written: {OUTPUT_CSV}")
    print(f"Subjects kept:     {len(rows)}")
    labels = Counter(r["label"] for r in rows)
    print(f"  CN: {labels.get('CN', 0)}    MCI: {labels.get('MCI', 0)}    AD: {labels.get('AD', 0)}")
    variants = Counter(r["variant"] for r in rows)
    print(f"  variant used -> {dict(variants)}")
    fs = Counter((r["field_strength"] or "?") for r in rows)
    print(f"  field strength -> {dict(fs)}   (expect all 1.5)")
    no_dcm = [r["subject_id"] for r in rows if r["n_dcm"] == 0]
    if no_dcm:
        print(f"  subjects with 0 DICOMs: {no_dcm}")
    print("=" * 60)

    if flags:
        print(f"\n{len(flags)} subject(s) need attention:")
        for line in flags:
            print("  -", line)
    else:
        print("\nNo problems flagged. Every subject mapped to a DICOM folder.")


if __name__ == "__main__":
    main()
