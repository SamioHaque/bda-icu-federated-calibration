"""Generate paper tables from the S4 metrics CSV.

LOCAL / DRIVER-SIDE ONLY -- operates on a ~1k-row result table already
produced by the distributed pipeline. Standard library only, so it runs
anywhere without installing anything.

Usage (from repo root):
    python scripts\\report.py
"""
import csv
import glob
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERN = os.path.join(ROOT, "results", "metrics", "metrics", "*.csv")

NUM = ("n", "n_pos", "n_neg", "prevalence", "auroc", "auprc", "brier",
       "mean_p", "ece", "mce", "threshold", "tp", "fp", "fn",
       "precision", "recall", "f1")


def load():
    files = [f for f in glob.glob(PATTERN) if not os.path.basename(f).startswith("_")]
    if not files:
        sys.exit("No metrics CSV under %s\nRun: .\\scripts\\pull_results.ps1" % PATTERN)
    rows = []
    for fp in files:
        with open(fp, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                for k in NUM:
                    if k in r and r[k] not in (None, "", "NULL"):
                        try:
                            r[k] = float(r[k])
                        except ValueError:
                            pass
                rows.append(r)
    return [r for r in rows if r.get("split") == "test"]


def ms(vals, nd=4):
    """mean +/- sd, or mean alone when a single seed."""
    vals = [v for v in vals if isinstance(v, float)]
    if not vals:
        return "--"
    if len(vals) == 1:
        return "%.*f" % (nd, vals[0])
    return "%.*f+-%.*f" % (nd, st.mean(vals), nd, st.pstdev(vals))


def group(rows, keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def line(w):
    print("-" * w)


def t1(rows):
    print("\n=== TABLE 1: pooled test performance (eval_client = ALL) ===")
    print("mean +- sd across seeds; n_seeds in brackets\n")
    sel = [r for r in rows if r["eval_client"] == "ALL"]
    g = group(sel, ["model", "regime", "train_client", "condition", "calibration"])
    hdr = "%-6s %-14s %-14s %-9s %-9s %-16s %-16s %-16s %-16s" % (
        "model", "regime", "train_client", "cond", "calib",
        "AUROC", "AUPRC", "Brier", "ECE")
    print(hdr); line(len(hdr))
    for k in sorted(g):
        v = g[k]
        print("%-6s %-14s %-14s %-9s %-9s %-16s %-16s %-16s %-16s [%d]" % (
            k[0], k[1], k[2], k[3], k[4],
            ms([r["auroc"] for r in v]), ms([r["auprc"] for r in v]),
            ms([r["brier"] for r in v], 5), ms([r["ece"] for r in v], 5), len(v)))


def t2(rows):
    print("\n=== TABLE 2: effect of isotonic calibration (eval_client = ALL) ===")
    print("ECE and Brier before/after; dAUROC shows discrimination is preserved\n")
    sel = [r for r in rows if r["eval_client"] == "ALL"]
    g = group(sel, ["model", "regime", "train_client", "condition"])
    hdr = "%-6s %-14s %-14s %-9s %-10s %-10s %-8s %-10s %-10s %-10s" % (
        "model", "regime", "train_client", "cond",
        "ECE_raw", "ECE_iso", "ECE_red", "Brier_raw", "Brier_iso", "dAUROC")
    print(hdr); line(len(hdr))
    worst = []
    for k in sorted(g):
        v = g[k]
        raw = [r for r in v if r["calibration"] == "raw"]
        iso = [r for r in v if r["calibration"] == "isotonic"]
        if not raw or not iso:
            continue
        er = st.mean([r["ece"] for r in raw]); ei = st.mean([r["ece"] for r in iso])
        br = st.mean([r["brier"] for r in raw]); bi = st.mean([r["brier"] for r in iso])
        ar = st.mean([r["auroc"] for r in raw]); ai = st.mean([r["auroc"] for r in iso])
        red = 100.0 * (er - ei) / er if er > 0 else 0.0
        worst.append((er, k))
        print("%-6s %-14s %-14s %-9s %-10.5f %-10.5f %-7.1f%% %-10.5f %-10.5f %+.4f" % (
            k[0], k[1], k[2], k[3], er, ei, red, br, bi, ai - ar))
    worst.sort(reverse=True)
    print("\nWorst raw ECE:")
    for er, k in worst[:5]:
        print("  %.5f  %s" % (er, " / ".join(k)))


def t3(rows, model, condition, calibration):
    print("\n=== TABLE 3: cross-client transfer, %s / %s / %s ===" % (
        model, condition, calibration))
    print("rows = training client, cols = evaluation client; diagonal = local\n")
    sel = [r for r in rows
           if r["model"] == model and r["condition"] == condition
           and r["calibration"] == calibration and r["regime"] == "R2_local"
           and r["eval_client"] != "ALL"]
    if not sel:
        print("  (no cells)"); return
    tr = sorted({r["train_client"] for r in sel})
    ev = sorted({r["eval_client"] for r in sel})
    for metric, nd in (("auroc", 4), ("ece", 5)):
        print("  [%s]" % metric.upper())
        print("    %-16s" % "train\\eval" + "".join("%-18s" % e for e in ev))
        for t in tr:
            cells = []
            for e in ev:
                v = [r[metric] for r in sel
                     if r["train_client"] == t and r["eval_client"] == e]
                cells.append("%-18s" % ms(v, nd))
            print("    %-16s" % t + "".join(cells))
        # off-diagonal degradation
        diag, off = [], []
        for r in sel:
            (diag if r["train_client"] == r["eval_client"] else off).append(r[metric])
        if diag and off:
            print("    diagonal mean %.*f | off-diagonal mean %.*f | delta %+.*f"
                  % (nd, st.mean(diag), nd, st.mean(off), nd,
                     st.mean(off) - st.mean(diag)))
        print()


def t4(rows):
    print("\n=== TABLE 4: federated aggregation strategies (eval_client = ALL) ===")
    sel = [r for r in rows if r["model"] == "lr_fl" and r["eval_client"] == "ALL"]
    if not sel:
        print("  (no FL rows -- has s3b_fedavg run?)"); return
    g = group(sel, ["regime", "condition", "calibration"])
    hdr = "%-16s %-9s %-9s %-16s %-16s %-16s %-16s" % (
        "regime", "cond", "calib", "AUROC", "AUPRC", "Brier", "ECE")
    print(hdr); line(len(hdr))
    for k in sorted(g):
        v = g[k]
        print("%-16s %-9s %-9s %-16s %-16s %-16s %-16s [%d]" % (
            k[0], k[1], k[2],
            ms([r["auroc"] for r in v]), ms([r["auprc"] for r in v]),
            ms([r["brier"] for r in v], 5), ms([r["ece"] for r in v], 5), len(v)))


def t5(rows):
    print("\n=== TABLE 5: per-client performance of the pooled model (R1) ===")
    print("does one global model serve every client equally?\n")
    sel = [r for r in rows if r["regime"] == "R1_pooled"
           and r["eval_client"] != "ALL" and r["calibration"] == "isotonic"]
    g = group(sel, ["model", "condition", "eval_client"])
    hdr = "%-6s %-9s %-16s %-16s %-16s %-16s" % (
        "model", "cond", "eval_client", "AUROC", "ECE", "prevalence")
    print(hdr); line(len(hdr))
    for k in sorted(g):
        v = g[k]
        print("%-6s %-9s %-16s %-16s %-16s %-16s" % (
            k[0], k[1], k[2],
            ms([r["auroc"] for r in v]), ms([r["ece"] for r in v], 5),
            ms([r["prevalence"] for r in v])))


def main():
    rows = load()
    print("Loaded %d test-split metric rows" % len(rows))
    t1(rows)
    t2(rows)
    for cond in ("careunit", "iid"):
        t3(rows, "gbt", cond, "raw")
    t3(rows, "gbt", "careunit", "isotonic")
    t4(rows)
    t5(rows)


if __name__ == "__main__":
    main()