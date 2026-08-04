"""Generate a SYNTHETIC illustrative feature matrix.

Purpose: show the schema and shape of the S2/S3 feature matrix in slides and
in the report appendix WITHOUT reproducing any MIMIC-IV record. Every value
here is drawn from a generator, not from the dataset. Marginals are set to
generic adult-ICU physiological ranges so the table reads plausibly; they are
NOT fitted to MIMIC and must never be quoted as results.

The real cohort is 65,366 stays at 10.84% mortality with 58 model features.
This file mimics the column layout only.

Usage:  python scripts\\make_synthetic_example.py
"""
import csv
import math
import os
import random

random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "synthetic")
os.makedirs(OUT, exist_ok=True)

N = 200

CONCEPTS = ["hr", "sbp", "dbp", "map", "rr", "spo2", "temp"]
AGGS = ["min", "max", "mean", "std", "first", "last", "count"]

# (centre, spread, plausible floor, plausible ceiling) -- generic adult ICU
DIST = {
    "hr":   (85.0, 18.0, 20.0, 300.0),
    "sbp":  (118.0, 20.0, 30.0, 300.0),
    "dbp":  (62.0, 13.0, 10.0, 200.0),
    "map":  (78.0, 13.0, 20.0, 250.0),
    "rr":   (19.0, 5.0, 0.0, 80.0),
    "spo2": (97.0, 2.5, 50.0, 100.0),
    "temp": (36.9, 0.6, 25.0, 45.0),
}
# Observed per-concept coverage in the real cohort, reproduced as missing rates.
COVERAGE = {"hr": 0.998, "sbp": 0.998, "dbp": 0.998, "map": 0.998,
            "rr": 0.998, "spo2": 0.998, "temp": 0.969}
GROUPS = [("cardiac_neuro", 0.392), ("medical", 0.346), ("surgical", 0.262)]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def concept_block(c):
    """One concept's seven aggregates, internally consistent (min<=mean<=max)."""
    centre, spread, lo, hi = DIST[c]
    if random.random() > COVERAGE[c]:
        return {a: "" for a in AGGS[:-1]} | {"count": 0}, 1

    mu = clamp(random.gauss(centre, spread * 0.6), lo, hi)
    sd = abs(random.gauss(spread * 0.35, spread * 0.15)) + 0.1
    n = max(1, int(abs(random.gauss(28, 12))))
    if c == "temp":
        n = max(1, int(abs(random.gauss(9, 4))))

    vals = [clamp(random.gauss(mu, sd), lo, hi) for _ in range(min(n, 60))]
    vals_sorted = sorted(vals)
    out = {
        "min": round(vals_sorted[0], 2),
        "max": round(vals_sorted[-1], 2),
        "mean": round(sum(vals) / len(vals), 2),
        "std": round(sd, 2) if n > 1 else "",
        "first": round(vals[0], 2),
        "last": round(vals[-1], 2),
        "count": n,
    }
    return out, 0


def main():
    cols = ["stay_id", "subject_id", "client_group", "split",
            "anchor_age", "gender"]
    cols += ["%s_%s" % (c, a) for c in CONCEPTS for a in AGGS]
    cols += ["%s_missing" % c for c in CONCEPTS]
    cols += ["hospital_expire_flag"]

    rows = []
    for i in range(N):
        r = {}
        r["stay_id"] = 30000000 + i * 137          # fabricated, not MIMIC ids
        r["subject_id"] = 10000000 + i * 91
        u = random.random()
        acc = 0.0
        for g, w in GROUPS:
            acc += w
            if u <= acc:
                r["client_group"] = g
                break
        r["split"] = random.choices(["train", "val", "test"], [60, 20, 20])[0]
        r["anchor_age"] = random.randint(18, 91)
        r["gender"] = random.choice(["M", "F"])

        for c in CONCEPTS:
            blk, miss = concept_block(c)
            for a in AGGS:
                r["%s_%s" % (c, a)] = blk[a]
            r["%s_missing" % c] = miss

        # Plausible risk direction only, so the table does not look random.
        def g(k, d):
            v = r.get(k, "")
            return float(v) if v != "" else d
        z = (-1.97
             + 0.020 * (g("hr_mean", 85) - 85)
             - 0.014 * (g("sbp_mean", 118) - 118)
             + 0.060 * (g("rr_mean", 19) - 19)
             - 0.090 * (g("spo2_min", 95) - 95)
             + 0.022 * (r["anchor_age"] - 65))
        # Intercept chosen so synthetic prevalence lands near the real
        # cohort's 10.84%, purely so the illustration is not misleading.
        p = 1.0 / (1.0 + math.exp(-z))
        r["hospital_expire_flag"] = 1 if random.random() < p else 0
        rows.append(r)

    full = os.path.join(OUT, "SYNTHETIC_feature_matrix_200rows.csv")
    with open(full, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Narrow excerpt: 58 columns will not fit on a slide.
    slim_cols = ["stay_id", "client_group", "anchor_age", "gender",
                 "hr_mean", "hr_min", "hr_max", "hr_count",
                 "sbp_mean", "spo2_min", "temp_mean", "temp_missing",
                 "hospital_expire_flag"]
    slim = os.path.join(OUT, "SYNTHETIC_slide_excerpt_10rows.csv")
    with open(slim, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=slim_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows[:10])

    n_pos = sum(r["hospital_expire_flag"] for r in rows)
    print("wrote %s  (%d rows x %d cols)" % (full, len(rows), len(cols)))
    print("wrote %s  (10 rows x %d cols)" % (slim, len(slim_cols)))
    print("synthetic prevalence: %d/%d = %.3f  (real cohort: 0.1084)"
          % (n_pos, len(rows), n_pos / len(rows)))
    print("\nSYNTHETIC DATA -- illustrative schema only, not MIMIC-IV records.")


if __name__ == "__main__":
    main()