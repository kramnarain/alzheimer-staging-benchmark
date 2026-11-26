"""
Regenerate the figures and table values from the 5-fold cross-validation results
(the cv_folds_long_*.csv written by 09_train_cv.py). Produces the PNGs and prints the
table text (5-fold mean +/- SD) plus paired across-fold significance. No GPU or model
needed.

Figures written to ./figures :
  fig2_within_adni.png        within-cohort metrics, ADNI
  fig3_within_oasis.png       within-cohort metrics, OASIS-3
  fig4_cross_heatmap.png      cross-dataset generalisation heatmap (per model)
  fig5_within_vs_cross.png    within vs cross balanced accuracy and gap (per model)

Set SRC to the cv_folds_long CSV to plot.
"""
import os, csv, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "cv_folds_long.csv")   # set to your 5-fold results CSV
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

MODELS = ["convnext", "mambad", "vil", "vrwkv"]
MLABEL = {"convnext": "ConvNeXt", "mambad": "MambAD", "vil": "Vision-xLSTM", "vrwkv": "Vision-RWKV"}
CLABEL = {"adni": "ADNI", "oasis": "OASIS-3"}
CHANCE = 1/3

rows = [r for r in csv.DictReader(open(SRC)) if r["level"] == "subject"]

def folds(train, test, metric):
    d = defaultdict(dict)
    for r in rows:
        if r["train_cohort"] == train and r["test_cohort"] == test and r["eval"] in ("within", "cross"):
            v = r[metric]
            if v is None or v == "" or v.lower() == "nan":
                continue
            d[r["model"]][int(r["fold"])] = float(v)
    return {m: [d[m][k] for k in sorted(d[m])] for m in d}

def ms(vals):
    a = np.array(vals); return a.mean(), a.std(ddof=1)

# ---------------------------------------------------------------- figures 2 & 3
METRICS = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced acc."),
           ("f1_macro", "Macro-F1"), ("macro_auc", "Macro-AUC"), ("mcc", "MCC")]
def within_fig(cohort, outfile):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(METRICS)); w = 0.2
    for i, m in enumerate(MODELS):
        means, sds = [], []
        for key, _ in METRICS:
            f = folds(cohort, cohort, key)[m]; mu, sd = ms(f); means.append(mu); sds.append(sd)
        ax.bar(x + (i-1.5)*w, means, w, yerr=sds, capsize=2, label=MLABEL[m])
    ax.axhline(CHANCE, ls="--", c="0.4", lw=1)
    ax.text(len(METRICS)-0.5, CHANCE+0.01, "chance (1/3)", c="0.4", fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in METRICS])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.set_title(f"Within-cohort performance on {CLABEL[cohort]} (5-fold mean, error bars = SD)")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(outfile, dpi=200, bbox_inches="tight"); plt.close(fig)

within_fig("adni", os.path.join(FIGDIR, "fig2_within_adni.png"))
within_fig("oasis", os.path.join(FIGDIR, "fig3_within_oasis.png"))

# ---------------------------------------------------------------- figure 4 heatmap
fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
cohorts = ["adni", "oasis"]
for k, (ax, m) in enumerate(zip(axes, MODELS)):
    M = np.zeros((2, 2))
    for i, tr in enumerate(cohorts):
        for j, te in enumerate(cohorts):
            M[i, j] = ms(folds(tr, te, "balanced_accuracy")[m])[0]
    im = ax.imshow(M, vmin=0.33, vmax=0.55, cmap="viridis")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.47 else "black", fontsize=10)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, ec="red", lw=2))
    ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False, ec="red", lw=2))
    ax.set_xticks([0, 1]); ax.set_xticklabels([CLABEL[c] for c in cohorts], fontsize=8)
    ax.set_yticks([0, 1])
    if k == 0:                                   # y labels only on the leftmost panel
        ax.set_yticklabels([CLABEL[c] for c in cohorts], fontsize=8)
        ax.set_ylabel("Trained on")
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Tested on"); ax.set_title(MLABEL[m], fontsize=10)
fig.subplots_adjust(wspace=0.25)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Balanced accuracy")
fig.suptitle("Cross-dataset generalisation (subject-level balanced accuracy); diagonal = within-cohort", fontsize=10)
fig.savefig(os.path.join(FIGDIR, "fig4_cross_heatmap.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- figure 5 within vs cross
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(MODELS)); w = 0.35
win, cro = [], []
for m in MODELS:
    wv = ms(folds("adni", "adni", "balanced_accuracy")[m])[0], ms(folds("oasis", "oasis", "balanced_accuracy")[m])[0]
    cv = ms(folds("adni", "oasis", "balanced_accuracy")[m])[0], ms(folds("oasis", "adni", "balanced_accuracy")[m])[0]
    win.append(np.mean(wv)); cro.append(np.mean(cv))
ax.bar(x - w/2, win, w, label="Within-cohort (mean)")
ax.bar(x + w/2, cro, w, label="Cross-cohort (mean)")
for i in range(len(MODELS)):
    ax.annotate(f"gap {win[i]-cro[i]:+.3f}", (x[i], max(win[i], cro[i])+0.01), ha="center", fontsize=8)
ax.axhline(CHANCE, ls="--", c="0.4", lw=1)
ax.set_xticks(x); ax.set_xticklabels([MLABEL[m] for m in MODELS])
ax.set_ylabel("Balanced accuracy"); ax.set_ylim(0, 0.6)
ax.set_title("Mean within-cohort vs cross-cohort balanced accuracy, with the generalisation gap")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig5_within_vs_cross.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

print("figures written to", os.path.abspath(FIGDIR))
for f in sorted(os.listdir(FIGDIR)):
    print("  ", f)

# ---------------------------------------------------------------- table digests
def line(train, test, m):
    out = {}
    for key in ["balanced_accuracy", "sensitivity_macro", "specificity_macro", "f1_macro",
                "mcc", "cohen_kappa", "macro_auc", "pr_auc_macro", "accuracy"]:
        mu, sd = ms(folds(train, test, key)[m]); out[key] = f"{mu:.3f} +/- {sd:.3f}"
    return out

print("\n================= WITHIN-COHORT (subject-level, 5-fold mean +/- SD) =================")
print(f"{'cohort':7}{'model':12}{'bal_acc':16}{'sens':16}{'spec':16}{'F1':16}{'acc':16}")
for c in ["adni", "oasis"]:
    for m in MODELS:
        d = line(c, c, m)
        print(f"{CLABEL[c]:7}{MLABEL[m]:12}{d['balanced_accuracy']:16}{d['sensitivity_macro']:16}{d['specificity_macro']:16}{d['f1_macro']:16}{d['accuracy']:16}")

print("\n================= WITHIN-COHORT (probabilistic / agreement metrics) =================")
print(f"{'cohort':7}{'model':12}{'macroAUC':16}{'PR-AUC':16}{'MCC':16}{'kappa':16}")
for c in ["adni", "oasis"]:
    for m in MODELS:
        d = line(c, c, m)
        print(f"{CLABEL[c]:7}{MLABEL[m]:12}{d['macro_auc']:16}{d['pr_auc_macro']:16}{d['mcc']:16}{d['cohen_kappa']:16}")

print("\n================= CROSS-COHORT MATRIX (subject-level balanced accuracy) =================")
print(f"{'model':12}{'A->A':16}{'A->O':16}{'O->A':16}{'O->O':16}")
for m in MODELS:
    aa = line('adni','adni',m)['balanced_accuracy']; ao = line('adni','oasis',m)['balanced_accuracy']
    oa = line('oasis','adni',m)['balanced_accuracy']; oo = line('oasis','oasis',m)['balanced_accuracy']
    print(f"{MLABEL[m]:12}{aa:16}{ao:16}{oa:16}{oo:16}")

print("\n================= WITHIN vs CROSS (mean and gap) =================")
print(f"{'model':12}{'within':10}{'cross':10}{'gap':8}")
for i, m in enumerate(MODELS):
    print(f"{MLABEL[m]:12}{win[i]:<10.3f}{cro[i]:<10.3f}{win[i]-cro[i]:<+8.3f}")

print("\n================= PAIRED ACROSS-FOLD SIGNIFICANCE (within-cohort balanced accuracy) =================")
import itertools
def folds_map(train, test, metric):
    d = defaultdict(dict)
    for r in rows:
        if r["train_cohort"] == train and r["test_cohort"] == test:
            v = r[metric]
            if v not in (None, "") and v.lower() != "nan":
                d[r["model"]][int(r["fold"])] = float(v)
    return d
for c in ["adni", "oasis"]:
    print(f"-- {CLABEL[c]} -- paired t-test across folds, Holm-corrected (6 pairs)")
    fm = folds_map(c, c, "balanced_accuracy")
    pairs = list(itertools.combinations(MODELS, 2)); raw = []
    for a, b in pairs:
        common = sorted(set(fm[a]) & set(fm[b]))
        fa = [fm[a][k] for k in common]; fb = [fm[b][k] for k in common]
        t, p = stats.ttest_rel(fa, fb); raw.append((a, b, p))
    # Holm
    order = sorted(range(len(raw)), key=lambda i: raw[i][2]); adj = [None]*len(raw); k = len(raw)
    for rank, i in enumerate(order):
        adj[i] = min(1.0, raw[i][2]*(k-rank))
    for (a, b, p), pa in zip(raw, adj):
        print(f"   {MLABEL[a]:12} vs {MLABEL[b]:12} p_raw={p:.3f}  p_Holm={pa:.3f}  {'*' if pa<0.05 else 'ns'}")
