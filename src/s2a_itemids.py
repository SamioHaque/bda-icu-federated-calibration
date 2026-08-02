"""S2a: verify chartevents itemids against d_items before the expensive scan.

d_items is ~4k rows, so this is seconds of work. A wrong or renamed itemid is
caught here rather than after a 433M-row aggregation produces an all-null column.
"""
import argparse
from pyspark.sql import SparkSession, functions as F

import config as C
from manifest import Run

# Candidate ids with the label each is expected to carry. Asserted at runtime;
# never trusted blind. 220045/220050/220277 are corroborated by the prior pipeline.
CANDIDATES = {
    220045: "Heart Rate",
    220050: "Arterial Blood Pressure systolic",
    220051: "Arterial Blood Pressure diastolic",
    220052: "Arterial Blood Pressure mean",
    220179: "Non Invasive Blood Pressure systolic",
    220180: "Non Invasive Blood Pressure diastolic",
    220181: "Non Invasive Blood Pressure mean",
    220210: "Respiratory Rate",
    220277: "O2 saturation pulseoxymetry",
    223761: "Temperature Fahrenheit",
    223762: "Temperature Celsius",
    220739: "GCS - Eye Opening",
    223900: "GCS - Verbal Response",
    223901: "GCS - Motor Response",
}

# Independent label search, so we discover ids the candidate list missed
# (e.g. alternate respiratory-rate or temperature routes).
SEARCH = {
    "heart_rate":  r"heart rate",
    "sbp":         r"blood pressure systolic",
    "dbp":         r"blood pressure diastolic",
    "map":         r"pressure mean",
    "resp_rate":   r"respiratory rate",
    "spo2":        r"o2 saturation|spo2",
    "temperature": r"temperature",
    "gcs":         r"gcs",
}


def read_bq(spark, table):
    return (spark.read.format("bigquery")
            .option("table", f"{C.PROJECT_ID}.{C.DATASET}.{table}")
            .option("parentProject", C.PROJECT_ID)
            .load())


def main(args):
    spark = SparkSession.builder.appName("s2a_itemids").getOrCreate()
    spark.conf.set("temporaryGcsBucket", C.BUCKET)
    run = Run(spark, "s2a_itemids", args.git_sha,
              {"n_candidates": len(CANDIDATES)})
    run.await_executors(args.expect_executors)

    try:
        d = read_bq(spark, "d_items")
        print("d_items columns:", d.columns)
        run.count("d_items_rows", d.count())

        cols = [c for c in ["itemid", "label", "abbreviation", "linksto",
                            "category", "unitname", "param_type",
                            "lownormalvalue", "highnormalvalue"]
                if c in d.columns]

        # --- 1. Do the candidate ids exist, and do labels match expectation?
        cand = (d.filter(F.col("itemid").isin(list(CANDIDATES.keys())))
                 .select(*cols).orderBy("itemid"))
        print("\n=== CANDIDATE ITEMIDS (as recorded in d_items) ===")
        cand.show(50, truncate=False)

        found = {r["itemid"]: r["label"] for r in cand.select("itemid", "label").collect()}
        missing   = sorted(set(CANDIDATES) - set(found))
        mismatched = {i: {"expected": CANDIDATES[i], "actual": found[i]}
                      for i in found if found[i].strip().lower() != CANDIDATES[i].lower()}

        run.param("missing_itemids", missing)
        run.param("label_mismatches", mismatched)
        if missing:
            print(f"\n[WARN] itemids absent from d_items: {missing}")
        if mismatched:
            print(f"\n[WARN] label mismatches: {mismatched}")

        # --- 2. Independent label search: surface ids we did not anticipate
        print("\n=== LABEL SEARCH (chartevents only) ===")
        ce = d.filter(F.lower(F.col("linksto")) == "chartevents") \
              if "linksto" in d.columns else d
        discovered = {}
        for concept, pattern in SEARCH.items():
            hits = (ce.filter(F.lower(F.col("label")).rlike(pattern))
                      .select(*cols).orderBy("itemid"))
            print(f"\n--- {concept}  /{pattern}/ ---")
            hits.show(30, truncate=False)
            discovered[concept] = [r.asDict() for r in
                                   hits.select("itemid", "label").limit(30).collect()]
        run.param("discovered", discovered)

        run.finish(C.PATH_MANIFEST, "ok")
    except Exception as e:
        run.finish(C.PATH_MANIFEST, "failed", e)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--git-sha", default="nogit")
    p.add_argument("--expect-executors", type=int, default=4)
    main(p.parse_args())