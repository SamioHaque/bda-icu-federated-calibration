"""Build the scalability table from local S5 manifests.

LOCAL ONLY -- reads results\\manifests\\s5_scalability-*.json. Standard library.

Usage (repo root):   python scripts\\scaling_table.py
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = os.path.join(ROOT, "results", "manifests", "s5_scalability-*.json")


def main():
    files = sorted(glob.glob(PAT))
    if not files:
        sys.exit("No S5 manifests at %s\nRun: .\\scripts\\pull_results.ps1" % PAT)

    recs = []
    for fp in files:
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if d.get("status") != "ok":
            print("SKIP (%s): %s" % (d.get("status"), os.path.basename(fp)))
            continue
        p = d.get("params", {})
        c = d.get("cluster", {})
        recs.append({
            "stage": p.get("stage"),
            "model": p.get("model") or "na",
            "frac": p.get("fraction"),
            "req": p.get("requested_executors"),
            "live": p.get("live_executors_measured", c.get("live_executors")),
            "cores": p.get("total_cores"),
            "measured": p.get("measured_s"),
            "wall": d.get("wall_clock_s"),
            "rows": (d.get("counts", {}).get("feature_rows")
                     or d.get("counts", {}).get("train_rows")),
            "shuffle": p.get("shuffle_partitions"),
        })

    print("Loaded %d successful S5 runs\n" % len(recs))

    # Aggregate repeats of the same sweep point. Single measurements cannot
    # distinguish a real scaling ceiling from one unlucky run (straggler, GC
    # pause, read-stream contention), so reps matter for the headline curve.
    cells = {}
    for r in recs:
        cells.setdefault((r["stage"], r["model"], r["frac"], r["req"]), []).append(r)

    def agg(vals):
        v = [x for x in vals if x is not None]
        if not v:
            return None, None, 0
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5 if len(v) > 1 else 0.0
        return m, sd, len(v)

    for stage in ("s2", "s3"):
        for model in ("na", "lr", "gbt"):
            keys = [k for k in cells if k[0] == stage and k[1] == model]
            if not keys:
                continue
            title = stage.upper() if model == "na" else "%s / %s" % (stage.upper(), model)
            print("=== %s ===" % title)
            hdr = "%-7s %-5s %-6s %-18s %-5s %-10s %-11s %-9s" % (
                "frac", "req", "cores", "measured_s", "n", "speedup", "efficiency", "rows")
            print(hdr); print("-" * len(hdr))
            for frac in sorted({k[2] for k in keys}):
                reqs = sorted({k[3] for k in keys if k[2] == frac})
                base = None
                for rq in reqs:
                    rows = cells[(stage, model, frac, rq)]
                    m, sd, n = agg([x["measured"] for x in rows])
                    if rq == 1 or base is None:
                        base = m
                    sp = base / m if m else None
                    eff = (sp / rq * 100.0) if (sp and rq) else None
                    bad = [x for x in rows if x["req"] != x["live"]]
                    warn = "  <-- live != requested" if bad else ""
                    cv = ("  cv=%.0f%%" % (100.0 * sd / m)) if (n > 1 and m) else ""
                    print("%-7s %-5s %-6s %-18s %-5s %-10s %-11s %-9s%s%s" % (
                        frac, rq, rows[0]["cores"],
                        "%.2f+-%.2f" % (m, sd) if n > 1 else "%.2f" % m,
                        n,
                        "%.2fx" % sp if sp else "--",
                        "%.0f%%" % eff if eff else "--",
                        rows[0]["rows"] if rows[0]["rows"] else "--", warn, cv))
                print()

    print("=== paste-ready summary (stage,model,frac,req,cores,mean_s,sd_s,n) ===")
    for k in sorted(cells, key=lambda k: (k[0], k[1], k[2] or 0, k[3] or 0)):
        rows = cells[k]
        m, sd, n = agg([x["measured"] for x in rows])
        print("%s,%s,%s,%s,%s,%.2f,%.2f,%d" % (
            k[0], k[1], k[2], k[3], rows[0]["cores"], m, sd, n))


if __name__ == "__main__":
    main()