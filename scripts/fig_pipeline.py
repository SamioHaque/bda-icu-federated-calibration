"""Methodological workflow diagram (Figure 0 / Methods).

LOCAL ONLY -- pure drawing, no data access. Counts are transcribed from the run
manifests; update them only if those runs are repeated.

Usage:  python scripts\\fig_pipeline.py
Output: results\\figures\\fig0_pipeline.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

CB = {"grey": "#9A9A9A", "blue": "#0072B2", "green": "#009E73",
      "orange": "#E69F00", "purple": "#CC79A7", "verm": "#D55E00",
      "sky": "#56B4E9", "black": "#000000"}

plt.rcParams.update({"font.size": 8, "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, ax = plt.subplots(figsize=(9.4, 11.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.axis("off")

X, W = 6, 54
CX = X + W / 2.0
NOTE_X = 63


def box(x, y, w, h, title, body, colour, lw=1.6, fs_t=9, fs_b=7.3, alpha=0.11):
    for face, lwv in ((colour, lw), ("none", lw)):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.55,rounding_size=1.1",
                                    linewidth=lwv, edgecolor=colour,
                                    facecolor=face,
                                    alpha=alpha if face != "none" else 1.0,
                                    zorder=2))
    ax.text(x + 1.7, y + h - 1.4, title, fontsize=fs_t, fontweight="bold",
            va="top", ha="left", color=colour, zorder=4)
    if body:
        ax.text(x + 1.7, y + h - 4.0, body, fontsize=fs_b, va="top", ha="left",
                color="#222222", zorder=4, linespacing=1.42)


def arrow(x1, y1, x2, y2, colour="#444444", lw=1.3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=lw, color=colour,
                                 zorder=1, shrinkA=0, shrinkB=0))


def note(y, text):
    ax.text(NOTE_X, y, text, fontsize=7.0, va="top", ha="left",
            color="#555555", linespacing=1.5, zorder=4)


box(X, 91, W, 6.5, "MIMIC-IV v3.1  (physionet-data, BigQuery)",
    "patients · admissions · icustays · chartevents · d_items", CB["grey"], alpha=0.16)
note(97.2, "Copied into the project dataset with bq cp\n"
           "(data locality). Credentialed under the\n"
           "PhysioNet DUA; raw rows never leave GCP.")

box(X, 81.5, W, 7.0, "S0  Cohort construction",
    "adults · first ICU stay per subject · mortality label\n"
    "leakage-prone columns never enter the schema", CB["blue"])
note(88.2, "65,366 stays = 65,366 subjects\n7,086 deaths (10.84% prevalence)\n"
           "excluded by design: los, outtime, dod,\nlast_careunit, rn, anchor_year")

box(X, 71.5, W, 7.5, "S1  Deterministic splits + client partitions",
    "murmur3 hash of subject_id -> 60/20/20 train/val/test\n"
    "IID (control)   |   Non-IID (care unit, 3 clients)", CB["blue"])
note(78.6, "Test set at natural prevalence.\nOne patient in exactly one partition.\n"
           "Non-IID mortality 6.9% / 11.2% / 15.1%")

box(X, 63.5, W, 6.0, "S2a  Itemid verification",
    "candidate itemids asserted against d_items before any scan", CB["blue"])
note(69.2, "18 itemids -> 7 concepts. Multi-route BP\n"
           "pooled, Fahrenheit converted; SaO2 and\nventilator parameters excluded.")

box(X, 52.0, W, 9.0, "S2  Distributed feature engineering  (Spark)",
    "column-pruned read -> itemid filter -> broadcast 24 h window join\n"
    "-> groupBy(stay_id, concept) -> aggregate -> pivot -> Parquet\n"
    "min / max / mean / std / first / last / count per concept",
    CB["blue"], lw=2.1, alpha=0.17)
note(60.6, "432,997,491 chartevents rows scanned\n"
           "10,736,029 measurements aggregated\n"
           "58 features: 49 aggregates, 7 missingness\n"
           "indicators, age, sex. Coverage 99.8%\n(96.9% temperature)")

box(X, 38.0, 25.5, 11.0, "S3  Spark MLlib",
    "LogisticRegression · GBTClassifier\n\n"
    "R1   pooled\nR2   client-local\nR2e  prediction ensemble", CB["green"], fs_b=7.1)
box(X + 28.5, 38.0, 25.5, 11.0, "S3b  Federated solver",
    "logistic regression (convex)\nMLP, 16 hidden (non-convex)\n40 rounds x 5 local epochs\n\n"
    "R3  central · FedAvg · FedProx\n      FedAdam · FedAdagrad", CB["orange"], fs_b=6.9)
note(48.8, "MLlib exposes no warm start, so the local solver\n"
           "is hand-written: gradients by distributed\n"
           "treeReduce, all clients in a single pass.\n\n"
           "Preprocessing constants come from per-client\n"
           "sufficient statistics only; no record crosses a\n"
           "client boundary. GBT cannot be parameter-\n"
           "averaged, so federated arms are LR only.")

box(X, 24.5, W, 10.0, "S4  Calibration + evaluation",
    "isotonic regression fitted on VALIDATION only, at two scopes:\n"
    "    iso_pooled  -> pooled validation set (central recalibration)\n"
    "    iso_target  -> evaluation client's own validation set\n"
    "AUROC · AUPRC · Brier · ECE · MCE · P/R/F1 at a val-selected threshold",
    CB["purple"], lw=2.1, alpha=0.17, fs_b=7.1)
note(32.6, "340 isotonic calibrators. Metrics in a few\n"
           "distributed passes: Mann-Whitney AUROC with\n"
           "average ranks; average precision from a\n"
           "cumulative window. 2,160 metric rows over\n"
           "3 seeds, including the cross-client matrix.")

box(X + 26, 14.0, 28, 7.0, "S5  Scalability sweep",
    "re-runs S2 and S3 under\n1/2/4/8 executors x 25%/100% cohort", CB["verm"], fs_b=7.1)
note(20.2, "40 timed runs, single cluster session.\n"
           "ETL 2.9x at 8 executors; MLlib fit\n0.72-0.79x (anti-scaling).")

box(X, 5.0, W, 6.0, "Tables and figures",
    "generated from manifests and metrics; no value typed by hand",
    CB["black"], alpha=0.05)

for ya, yb in [(91, 88.5), (81.5, 79.0), (71.5, 69.5), (63.5, 61.0)]:
    arrow(CX, ya, CX, yb)
arrow(CX, 52.0, X + 12.75, 49.0)
arrow(CX, 52.0, X + 41.25, 49.0)
arrow(X + 12.75, 38.0, CX - 6, 34.5)
arrow(X + 41.25, 38.0, CX + 6, 34.5)
arrow(20, 24.5, 20, 11.0)
arrow(46, 14.0, 46, 11.0)

ax.add_patch(FancyBboxPatch((NOTE_X - 0.8, 5.0), 35, 6.2,
                            boxstyle="round,pad=0.55,rounding_size=1.1",
                            linewidth=1.4, edgecolor=CB["sky"],
                            facecolor=CB["sky"], alpha=0.13, zorder=2))
ax.text(NOTE_X + 0.8, 10.4, "Run manifest, written by every stage", fontsize=8.0,
        fontweight="bold", color=CB["sky"], va="top")
ax.text(NOTE_X + 0.8, 8.2, "git SHA · seed · executors · cores · shuffle\n"
                           "partitions · row counts · wall clock",
        fontsize=6.9, va="top", color="#222222", linespacing=1.42)

ax.text(50, 100.6, "Spark-based distributed EHR modelling for calibrated ICU risk "
                   "under non-IID patient distributions",
        fontsize=10.2, fontweight="bold", ha="center", va="top")

out = os.path.join(FIG, "fig0_pipeline.png")
fig.savefig(out)
plt.close(fig)
print("wrote %s" % out)