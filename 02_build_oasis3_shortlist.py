"""
Build a BALANCED OASIS-3 download shortlist from the CDR table.

Updated matching (maximises the AD class). OASIS-3 is longitudinal, so instead of
labelling each subject's single baseline scan, every MR session is considered and
each subject is assigned the MOST SEVERE stage (CN < MCI < AD) that one of its scans
reaches within MATCH_WINDOW_DAYS of a CDR assessment; the scan used is the one
closest in time to that assessment. This keeps one scan per subject (subject-level,
leakage-safe) while pairing an AD subject to the scan taken nearest its AD diagnosis,
which raises the usable AD count from ~107 (baseline-only) to ~205.

Steps:
  1. read oasis_mr_sessions.csv  -> all MR sessions per subject (subject + day + MR ID);
  2. read the UDSb4 CDR table (CDRTOT) -> CDR assessments per subject (day + CDR);
  3. for each subject, take the most severe stage reachable by a scan within
     MATCH_WINDOW_DAYS, using the scan with the smallest gap to that assessment;
  4. balance the classes (take TARGET_PER_CLASS, or all available AD if fewer),
     sampling reproducibly with SEED;
  5. write shortlist_experiment_ids.csv (MR IDs for the web "Download Images")
     and shortlist_labels.csv (subject, label, mr_id, gap_days) for the record.

CDR map: 0 -> CN, 0.5 -> MCI, >= 1 -> AD.
"""

import os, re, csv, random
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
DIR = "data/OASIS-3"          # folder holding the OASIS-3 clinical CSVs
MR_SESSIONS = os.path.join(DIR, "oasis_mr_sessions.csv")   # ALL MR sessions (NITRC-IR export)
CDR_CSV     = os.path.join(DIR, "OASIS3_UDSb4_cdr.csv")
OUT_IDS     = os.path.join(DIR, "shortlist_experiment_ids.csv")
OUT_LABELS  = os.path.join(DIR, "shortlist_labels.csv")

TARGET_PER_CLASS  = 205      # = all available AD under this scheme; CN/MCI matched down to it
MATCH_WINDOW_DAYS = 730
SEED = 42
# ---------------------------------------------------------------------------

SUBJ_RE = re.compile(r"(OAS3\d{4,5})", re.IGNORECASE)
MRID_RE = re.compile(r"(OAS3\d{4,5})_MR_d(\d+)", re.IGNORECASE)
DAY_RE  = re.compile(r"[_\-]?d(\d{3,5})", re.IGNORECASE)
ORDER   = {"CN": 0, "MCI": 1, "AD": 2}


def label_from_cdr(v):
    try: v = float(v)
    except (TypeError, ValueError): return None
    if v == 0: return "CN"
    if v == 0.5: return "MCI"
    if v >= 1: return "AD"
    return None


def read_mr_sessions(path):
    """subject -> list of (day, mr_id) across ALL MR sessions."""
    out = defaultdict(list)
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        mrcol = None
        for c in r.fieldnames:
            if c and c.strip().lower().replace("_", " ") in ("mr id", "mrid", "experiment id"):
                mrcol = c; break
        if mrcol is None and rows:
            mrcol = max(rows[0].keys(),
                        key=lambda c: sum(1 for row in rows if MRID_RE.search(str(row.get(c, "")))))
        for row in rows:
            m = MRID_RE.search(str(row.get(mrcol, "")))
            if m:
                out[m.group(1).upper()].append((int(m.group(2)), f"{m.group(1)}_MR_d{m.group(2)}"))
    return out


def read_cdr(path):
    """subject -> list of (day, label) from the UDSb4 CDR table."""
    out = defaultdict(list)
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in r.fieldnames}
        id_col  = cols.get("oasisid") or cols.get("oasis_session_label")
        day_col = cols.get("days_to_visit")
        lab_col = cols.get("oasis_session_label")
        cdr_col = cols.get("cdrtot") or cols.get("cdrglob") or cols.get("cdr")
        for row in r:
            m = SUBJ_RE.search(str(row.get(id_col, "")))
            if not m: continue
            day = None
            if day_col:
                try: day = int(float(row.get(day_col)))
                except (TypeError, ValueError): day = None
            if day is None and lab_col:
                dm = DAY_RE.search(str(row.get(lab_col, ""))); day = int(dm.group(1)) if dm else None
            lab = label_from_cdr(row.get(cdr_col))
            if lab and day is not None:
                out[m.group(1).upper()].append((day, lab))
    return out


def main():
    mr  = read_mr_sessions(MR_SESSIONS)
    cdr = read_cdr(CDR_CSV)
    print(f"  {len(mr)} subjects with MR sessions, {len(cdr)} with CDR")

    chosen = {}   # subject -> (label, mr_id, gap_days)
    for subj in set(mr) & set(cdr):
        cand = []  # (gap, label, mr_id)
        for day, mr_id in mr[subj]:
            for cday, clab in cdr[subj]:
                gap = abs(day - cday)
                if gap <= MATCH_WINDOW_DAYS:
                    cand.append((gap, clab, mr_id))
        if not cand: continue
        sev = max(ORDER[c[1]] for c in cand)                     # most severe stage reached
        best = min((c for c in cand if ORDER[c[1]] == sev), key=lambda c: c[0])   # closest scan
        chosen[subj] = (best[1], best[2], best[0])

    counts = Counter(v[0] for v in chosen.values())
    print(f"  available: CN {counts.get('CN',0)}  MCI {counts.get('MCI',0)}  AD {counts.get('AD',0)}")

    cap = min(TARGET_PER_CLASS, counts.get("AD", 0))             # AD is the binding class
    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for subj, (lab, mr_id, gap) in chosen.items():
        by_class[lab].append((subj, mr_id, gap))
    selected = []
    for lab in ("CN", "MCI", "AD"):
        pool = by_class[lab]; rng.shuffle(pool)
        take = pool[:cap]
        selected += [(s, lab, mid, gap) for (s, mid, gap) in take]
        print(f"  {lab}: selected {len(take)} of {len(pool)}")

    selected.sort(key=lambda t: t[0])
    with open(OUT_IDS, "w", encoding="utf-8", newline="\n") as f:
        f.write("experiment_id\n")
        for s, lab, mid, gap in selected:
            f.write(mid + "\n")
    with open(OUT_LABELS, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["subject_id", "label", "mr_id", "gap_days"])
        for s, lab, mid, gap in selected:
            w.writerow([s, lab, mid, gap])

    print(f"\n  shortlist: {len(selected)} subjects ({dict(Counter(t[1] for t in selected))})")
    print(f"  wrote {OUT_IDS}")
    print(f"  wrote {OUT_LABELS}")


if __name__ == "__main__":
    main()
