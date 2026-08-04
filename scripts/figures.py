"""Generate publication figures from local result files.

LOCAL / DRIVER-SIDE ONLY -- every input is a small result table already
produced by the distributed pipeline. Requires matplotlib only.

Usage (repo root):   python scripts\\figures.py
Outputs:             results\\figures\\*.png  (300 dpi)
"""
import csv
import glob
import json
import os
import statistics as st
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib required:  pip install matplotlib")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito: colourblind-safe
CB = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
      "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "verm": "#D55E00", "purple": "#CC79A7"}
SERIES = [CB["blue"], CB["verm"], CB["green"], CB["orange"], CB["purple"], CB["sky"]]

# The paper's terminology. "careunit" is the non-IID partition; "iid" is the
# control. Figures must use the paper's words, not the code's column values.
COND = {"careunit": "Non-IID (care unit)", "iid": "IID (control)"}


def cond_label(c):
    return COND.get(c, c)


plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.bbox": "tight",
})

FLOATS = ("auroc", "auprc", "brier", "ece", "mce", "prevalence", "mean_p",
          "precision", "recall", "f1", "threshold", "bin_conf", "bin_freq",
          "gap", "train_loss", "val_auroc", "val_logloss", "delta_norm",
          "elapsed_s")
INTS = ("n", "n_pos", "n_neg", "tp", "fp", "fn", "bin", "bin_n", "round", "seed")


def read_csv_dir(*parts):
    pat = os.path.join(RES, *parts, "*.csv")
    files = [f for f in glob.glob(pat) if not os.path.basename(f).startswith("_")]
    rows = []
    for fp in files:
        with open(fp, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                for k in FLOATS:
                    if r.get(k) not in (None, "", "NULL"):
                        try: r[k] = float(r[k])
                        except ValueError: pass
                for k in INTS:
                    if r.get(k) not in (None, "", "NULL"):
                        try: r[k] = int(float(r[k]))
                        except ValueError: pass
                rows.append(r)
    if not rows:
        print("  [warn] nothing at %s" % pat)
    return rows


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("  wrote %s" % path)


# --------------------------------------------------------------- Figure 1
def fig_reliability(rel):
    """Reliability diagrams for the worst transfer cell, one panel per scope."""
    scopes = [s for s in ("raw", "iso_pooled", "iso_target")
              if any(r["calibration"] == s for r in rel)]
    if not scopes:
        return
    sel = [r for r in rel
           if r["split"] == "test" and r["model"] == "gbt"
           and r["condition"] == "careunit" and r["regime"] == "R2_local"
           and r["train_client"] == "medical" and r["eval_client"] == "cardiac_neuro"]
    if not sel:
        print("  [warn] fig1: target cell absent")
        return

    fig, axes = plt.subplots(1, len(scopes), figsize=(3.0 * len(scopes), 3.2),
                             sharex=True, sharey=True)
    if len(scopes) == 1:
        axes = [axes]
    for ax, sc in zip(axes, scopes):
        s = [r for r in sel if r["calibration"] == sc]

        # ECE is computed PER SEED then averaged, matching the metrics tables.
        # Pooling bins across seeds first would let opposite-sign gaps cancel
        # and would report a smaller ECE than the tables do.
        per_seed = []
        for sd in sorted({r["seed"] for r in s}):
            ss = [r for r in s if r["seed"] == sd]
            tot = sum(r["bin_n"] for r in ss)
            if tot:
                per_seed.append(sum(r["gap"] * r["bin_n"] for r in ss) / tot)
        e = st.mean(per_seed) if per_seed else float("nan")

        bins = {}
        for r in s:
            bins.setdefault(r["bin"], []).append(r)
        xs, ys, ws = [], [], []
        for b in sorted(bins):
            v = bins[b]
            n = sum(x["bin_n"] for x in v)
            if n == 0:
                continue
            xs.append(sum(x["bin_conf"] * x["bin_n"] for x in v) / n)
            ys.append(sum(x["bin_freq"] * x["bin_n"] for x in v) / n)
            ws.append(n)
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, c=CB["black"], label="perfect")
        ax.plot(xs, ys, "o-", ms=4, lw=1.4, c=CB["blue"])
        smax = max(ws) if ws else 1
        ax.scatter(xs, ys, s=[40.0 * w / smax for w in ws], c=CB["blue"],
                   alpha=0.35, edgecolors="none")
        ax.set_title("%s\nECE = %.4f" % (sc, e), fontsize=9)
        ax.set_xlabel("predicted probability")
        ax.set_xlim(0, 0.55); ax.set_ylim(0, 0.55)
        ax.set_aspect("equal")
    axes[0].set_ylabel("observed frequency")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Non-IID transfer: GBT trained on medical, evaluated on "
                 "cardiac/neuro (prevalence 15.1% -> 6.9%)\n"
                 "below the diagonal = over-prediction of risk",
                 fontsize=9.5, y=1.06)
    save(fig, "fig1_reliability_transfer.png")


# --------------------------------------------------------------- Figure 2
def fig_transfer(met):
    """Transfer matrices: AUROC and ECE, raw vs locally recalibrated."""
    scopes = [s for s in ("raw", "iso_target") if any(r["calibration"] == s for r in met)]
    sel = [r for r in met
           if r["split"] == "test" and r["model"] == "gbt"
           and r["condition"] == "careunit" and r["regime"] == "R2_local"
           and r["eval_client"] != "ALL"]
    if not sel or not scopes:
        print("  [warn] fig2: no transfer cells")
        return
    tr = sorted({r["train_client"] for r in sel})
    ev = sorted({r["eval_client"] for r in sel})

    def cell(sc, metric, t, e):
        v = [r[metric] for r in sel
             if r["calibration"] == sc and r["train_client"] == t
             and r["eval_client"] == e and isinstance(r[metric], float)]
        return st.mean(v) if v else float("nan")

    # One colour scale per metric across ALL scopes. Independent scales would
    # make a 0.014 cell and a 0.105 cell render identically dark, implying no
    # improvement where there is a large one.
    scale = {}
    for metric in ("auroc", "ece"):
        vals = [cell(sc, metric, t, e) for sc in scopes for t in tr for e in ev]
        vals = [v for v in vals if v == v]
        scale[metric] = (min(vals), max(vals))

    fig, axes = plt.subplots(2, len(scopes), figsize=(3.5 * len(scopes), 6.6),
                             constrained_layout=True)
    if len(scopes) == 1:
        axes = axes.reshape(2, 1)
    for j, sc in enumerate(scopes):
        for i, (metric, cmap, fmt) in enumerate(
                [("auroc", "viridis", "%.3f"), ("ece", "magma_r", "%.4f")]):
            ax = axes[i][j]
            M = [[cell(sc, metric, t, e) for e in ev] for t in tr]
            vmin, vmax = scale[metric]
            im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(ev)))
            ax.set_xticklabels([e.replace("_", "\n") for e in ev], fontsize=7)
            ax.set_yticks(range(len(tr)))
            ax.set_yticklabels([t.replace("_", "\n") for t in tr], fontsize=7)
            mid = vmin + 0.55 * (vmax - vmin)
            for a in range(len(tr)):
                for b in range(len(ev)):
                    val = M[a][b]
                    if val != val:
                        continue
                    dark = (val > mid) if metric == "ece" else (val < mid)
                    ax.text(b, a, fmt % val, ha="center", va="center",
                            fontsize=7.5, color="white" if dark else "black")
            ax.set_title("%s  (%s)" % (metric.upper(), sc), fontsize=9)
            ax.grid(False)
            if i == 1:
                ax.set_xlabel("evaluated on")
            if j == 0:
                ax.set_ylabel("trained on")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Cross-client transfer, GBT, Non-IID (care unit) partition\n"
                 "shared colour scale per row", fontsize=10)
    save(fig, "fig2_transfer_matrix.png")


# --------------------------------------------------------------- Figure 3
def fig_fl(hist):
    """FL convergence: validation AUROC and training loss vs round."""
    if not hist:
        print("  [warn] fig3: no fl_history")
        return
    conds = sorted({r["condition"] for r in hist})
    strats = sorted({r["strategy"] for r in hist})
    fig, axes = plt.subplots(2, len(conds), figsize=(4.0 * len(conds), 6.0),
                             sharex=True)
    if len(conds) == 1:
        axes = axes.reshape(2, 1)
    for j, cond in enumerate(conds):
        for i, (col, ylab) in enumerate([("val_auroc", "validation AUROC"),
                                         ("train_loss", "training log-loss")]):
            ax = axes[i][j]
            for k, s in enumerate(strats):
                pts = {}
                for r in hist:
                    if r["condition"] != cond or r["strategy"] != s:
                        continue
                    v = r.get(col)
                    if not isinstance(v, float):
                        continue
                    pts.setdefault(r["round"], []).append(v)
                if not pts:
                    continue
                xs = sorted(pts)
                ys = [st.mean(pts[x]) for x in xs]
                sd = [st.pstdev(pts[x]) if len(pts[x]) > 1 else 0.0 for x in xs]
                c = SERIES[k % len(SERIES)]
                ax.plot(xs, ys, "-", lw=1.4, c=c, label=s)
                ax.fill_between(xs, [y - e for y, e in zip(ys, sd)],
                                [y + e for y, e in zip(ys, sd)], color=c, alpha=0.15)
                # Faint per-seed traces: distinguishes genuine oscillation
                # from an artifact of seeds evaluated on different cadences.
                for sd_id in sorted({r["seed"] for r in hist}):
                    pl = [(r["round"], r[col]) for r in hist
                          if r["condition"] == cond and r["strategy"] == s
                          and r["seed"] == sd_id and isinstance(r.get(col), float)]
                    if len(pl) > 1:
                        pl.sort()
                        ax.plot([p[0] for p in pl], [p[1] for p in pl],
                                "-", lw=0.4, c=c, alpha=0.35)
            ax.set_ylabel(ylab if j == 0 else "")
            if i == 1:
                ax.set_xlabel("communication round")
            if i == 0:
                ax.set_title(cond_label(cond), fontsize=9.5)
    axes[0][0].legend(fontsize=7, loc="lower right", ncol=2)
    fig.suptitle("Federated convergence, mean +- sd over 3 seeds", fontsize=10, y=0.995)
    save(fig, "fig3_fl_convergence.png")


# --------------------------------------------------------------- Figure 4
def fig_scaling():
    """Speedup vs executors: ETL scales, MLlib training does not."""
    recs = []
    for fp in sorted(glob.glob(os.path.join(RES, "manifests", "s5_scalability-*.json"))):
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if d.get("status") != "ok":
            continue
        p = d.get("params", {})
        recs.append({"stage": p.get("stage"), "model": p.get("model") or "na",
                     "frac": p.get("fraction"), "req": p.get("requested_executors"),
                     "s": p.get("measured_s")})
    if not recs:
        print("  [warn] fig4: no S5 manifests")
        return

    def curve(stage, model, frac):
        cells = {}
        for r in recs:
            if r["stage"] == stage and r["model"] == model and r["frac"] == frac:
                cells.setdefault(r["req"], []).append(r["s"])
        if 1 not in cells:
            return None
        base = st.mean(cells[1])
        xs = sorted(cells)
        mu = [base / st.mean(cells[x]) for x in xs]
        lo, hi = [], []
        for x in xs:
            v = [base / t for t in cells[x]]
            lo.append(min(v)); hi.append(max(v))
        return xs, mu, lo, hi, {x: len(cells[x]) for x in xs}

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    ideal = [1, 2, 4, 8]
    for ax in axes:
        ax.plot(ideal, ideal, ls="--", lw=0.8, c=CB["black"], label="linear")
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xticks(ideal); ax.set_xticklabels(ideal)
        ax.set_yticks(ideal); ax.set_yticklabels(ideal)
        ax.set_xlabel("executors (2 cores each)")

    for k, frac in enumerate(sorted({r["frac"] for r in recs if r["stage"] == "s2"})):
        c = curve("s2", "na", frac)
        if not c:
            continue
        xs, mu, lo, hi, n = c
        col = SERIES[k]
        axes[0].plot(xs, mu, "o-", ms=4, lw=1.5, c=col,
                     label="%d%% cohort (n=%d)" % (int(frac * 100), max(n.values())))
        axes[0].fill_between(xs, lo, hi, color=col, alpha=0.18)
    axes[0].set_title("S2: feature build over 433M chartevents rows", fontsize=9.5)
    axes[0].set_ylabel("speedup vs 1 executor")
    axes[0].legend(fontsize=7)

    k = 0
    for model in ("lr", "gbt"):
        for frac in sorted({r["frac"] for r in recs if r["stage"] == "s3"}):
            c = curve("s3", model, frac)
            if not c:
                continue
            xs, mu, lo, hi, n = c
            axes[1].plot(xs, mu, "o-", ms=4, lw=1.5, c=SERIES[k],
                         label="%s, %d%%" % (model.upper(), int(frac * 100)))
            k += 1
    axes[1].set_title("S3: MLlib fit on 39k rows x 58 features", fontsize=9.5)
    axes[1].legend(fontsize=7)
    save(fig, "fig4_scalability.png")


# --------------------------------------------------------------- Figure 5
def fig_dissociation(met):
    """AUROC vs ECE: calibration moves one axis and not the other."""
    sel = [r for r in met
           if r["split"] == "test" and r["condition"] == "careunit"
           and r["eval_client"] != "ALL"
           and isinstance(r.get("auroc"), float) and isinstance(r.get("ece"), float)]
    scopes = [s for s in ("raw", "iso_pooled", "iso_target")
              if any(r["calibration"] == s for r in sel)]
    if not sel or not scopes:
        print("  [warn] fig5: nothing to plot")
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for k, sc in enumerate(scopes):
        s = [r for r in sel if r["calibration"] == sc]
        ax.scatter([r["auroc"] for r in s], [r["ece"] for r in s],
                   s=14, alpha=0.65, c=SERIES[k], edgecolors="none", label=sc)
    ax.set_xlabel("AUROC (discrimination)")
    ax.set_ylabel("ECE (calibration error)")
    ax.set_title("Calibration shifts ECE, not AUROC\nNon-IID (care unit), all "
                 "regimes and clients", fontsize=9.5)
    ax.legend(fontsize=7.5)
    save(fig, "fig5_dissociation.png")


def fig_iid_contrast(met):
    """The title claim, in one panel: transfer penalty under IID vs Non-IID,
    before and after local recalibration."""
    scopes = [s for s in ("raw", "iso_target") if any(r["calibration"] == s for r in met)]
    if not scopes:
        return
    conds = [c for c in ("iid", "careunit") if any(r["condition"] == c for r in met)]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))

    for i, (metric, ylab) in enumerate([("ece", "ECE (calibration error)"),
                                        ("auroc", "AUROC (discrimination)")]):
        ax = axes[i]
        labels, width = [], 0.35
        for j, sc in enumerate(scopes):
            diag_v, off_v, xs = [], [], []
            for k, cond in enumerate(conds):
                s = [r for r in met
                     if r["split"] == "test" and r["model"] == "gbt"
                     and r["condition"] == cond and r["regime"] == "R2_local"
                     and r["eval_client"] != "ALL" and r["calibration"] == sc
                     and isinstance(r[metric], float)]
                d = [r[metric] for r in s if r["train_client"] == r["eval_client"]]
                o = [r[metric] for r in s if r["train_client"] != r["eval_client"]]
                if not d or not o:
                    continue
                diag_v.append(st.mean(d)); off_v.append(st.mean(o)); xs.append(k)
            if not xs:
                continue
            off = (j - (len(scopes) - 1) / 2.0) * width
            ax.bar([x + off - width / 4 for x in xs], diag_v, width / 2,
                   color=SERIES[j], alpha=0.55,
                   label="%s, local" % sc if i == 0 else None)
            ax.bar([x + off + width / 4 for x in xs], off_v, width / 2,
                   color=SERIES[j], alpha=1.0,
                   label="%s, transferred" % sc if i == 0 else None)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([cond_label(c) for c in conds], fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(metric.upper(), fontsize=9.5)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Transfer penalty is a Non-IID phenomenon; local "
                 "recalibration removes it from ECE but not AUROC",
                 fontsize=9.5, y=1.03)
    save(fig, "fig6_iid_vs_noniid.png")


def main():
    print("reading result tables ...")
    met = read_csv_dir("metrics", "metrics")
    rel = read_csv_dir("metrics", "reliability")
    hist = read_csv_dir("metrics", "fl_history")
    print("  metrics=%d  reliability=%d  fl_history=%d" % (len(met), len(rel), len(hist)))
    print("generating figures ...")
    fig_reliability(rel)
    fig_transfer(met)
    fig_fl(hist)
    fig_scaling()
    fig_dissociation(met)
    fig_iid_contrast(met)
    print("done -> %s" % FIG)


if __name__ == "__main__":
    main()