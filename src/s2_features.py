"""S2: first-24h vital aggregates from raw chartevents.

The distributed aggregation over ~433M raw event rows is this project's scale
claim. Features are built from chartevents, never from mimiciv_3_1_derived.

Design notes:
  - Column pruning at read time: 4 of ~15 columns, pushed down to the BQ
    Storage Read API, so wire volume is a fraction of the table.
  - The cohort (65k rows) is broadcast, so the 24h window join needs no shuffle
    of the large side.
  - Window bounds are compared in epoch seconds. intime is timestamp_ntz and
    interval arithmetic on that type is unreliable across connector versions.
  - first/last use min/max over struct(t, value) -- an aggregate, not a sort,
    so it costs no extra shuffle.
  - Missingness is an indicator per concept; rows are never dropped for
    incomplete vitals. That keeps the denominator fixed at S0's 65,366 and
    avoids the selection bias of a vitals-complete subset.
"""
import argparse
from pyspark.sql import SparkSession, functions as F

import config as C
from manifest import Run

CONCEPTS = list(C.CHART_CONCEPTS.keys())
ALL_ITEMIDS = sorted({i for v in C.CHART_CONCEPTS.values() for i in v})
WINDOW_S = 24 * 3600


def concept_col():
    """itemid -> concept name."""
    e = None
    for concept, ids in C.CHART_CONCEPTS.items():
        cond = F.col("itemid").isin(ids)
        e = F.when(cond, F.lit(concept)) if e is None else e.when(cond, F.lit(concept))
    return e.otherwise(F.lit(None))


def value_col():
    """valuenum, with Fahrenheit temperatures converted to Celsius."""
    f_to_c = (F.col("valuenum") - F.lit(32.0)) * F.lit(5.0) / F.lit(9.0)
    return F.when(F.col("itemid") == C.TEMP_F_ITEMID, f_to_c).otherwise(F.col("valuenum"))


def in_range_col():
    e = None
    for concept, bounds in C.CHART_RANGES.items():
        lo, hi = bounds
        cond = (F.col("concept") == concept) & (F.col("value") >= F.lit(lo)) & (F.col("value") <= F.lit(hi))
        e = F.when(cond, F.lit(True)) if e is None else e.when(cond, F.lit(True))
    return e.otherwise(F.lit(False))


def main(args):
    spark = SparkSession.builder.appName("s2_features").getOrCreate()
    spark.conf.set("temporaryGcsBucket", C.BUCKET)
    spark.conf.set("spark.sql.shuffle.partitions", str(args.shuffle))
    run = Run(spark, "s2_features", args.git_sha,
              {"seed": C.SEED, "window_hours": 24, "fraction": args.fraction,
               "shuffle_partitions": args.shuffle,
               "n_concepts": len(CONCEPTS), "n_itemids": len(ALL_ITEMIDS)})
    run.await_executors(args.expect_executors)

    try:
        cohort = spark.read.parquet(C.PATH_COHORT)
        cohort = cohort.select("subject_id", "stay_id", "intime", "hospital_expire_flag")

        # Data-fraction axis for the scalability sweep. Sampling on subject_id
        # keeps whole patients in or out.
        if args.fraction < 1.0:
            keep = int(round(args.fraction * 100))
            h = F.abs(F.hash(F.concat_ws("|", F.lit("frac"), F.col("subject_id")))) % 100
            cohort = cohort.filter(h < F.lit(keep))

        n_cohort = run.count("cohort_rows", cohort.count())

        win = cohort.select("stay_id", "intime")
        win = win.withColumn("t0", F.unix_timestamp("intime"))
        win = win.withColumn("t1", F.col("t0") + F.lit(WINDOW_S))
        win = win.select("stay_id", "t0", "t1")

        ce = (spark.read.format("bigquery")
              .option("table", f"{C.PROJECT_ID}.{C.DATASET}.chartevents")
              .option("parentProject", C.PROJECT_ID)
              .load()
              .select("stay_id", "itemid", "charttime", "valuenum"))

        # Metadata-only count: the paper's scale figure.
        run.count("chartevents_rows_total", ce.count())

        events = ce.filter(F.col("itemid").isin(ALL_ITEMIDS))
        events = events.filter(F.col("valuenum").isNotNull())
        events = events.join(F.broadcast(win), "stay_id")
        events = events.withColumn("t", F.unix_timestamp("charttime"))
        events = events.filter(F.col("t") >= F.col("t0"))
        events = events.filter(F.col("t") < F.col("t1"))
        events = events.withColumn("concept", concept_col())
        events = events.withColumn("value", value_col())
        events = events.filter(F.col("concept").isNotNull())
        events = events.withColumn("_ok", in_range_col())
        events = events.filter(F.col("_ok"))
        events = events.select("stay_id", "concept", "t", "value")

        long = events.groupBy("stay_id", "concept").agg(
            F.min("value").alias("min"),
            F.max("value").alias("max"),
            F.avg("value").alias("mean"),
            F.stddev_samp("value").alias("std"),
            F.min(F.struct("t", "value")).alias("_f"),
            F.max(F.struct("t", "value")).alias("_l"),
            F.count("value").alias("count"),
        )
        long = long.withColumn("first", F.col("_f.value"))
        long = long.withColumn("last", F.col("_l.value"))
        long = long.drop("_f", "_l").cache()

        run.count("measurements_used", long.agg(F.sum("count")).first()[0] or 0)
        run.count("stay_concept_pairs", long.count())

        cov = long.groupBy("concept").agg(
            F.countDistinct("stay_id").alias("stays"),
            F.sum("count").alias("measurements"),
        )
        cov = cov.withColumn("coverage", F.round(F.col("stays") / F.lit(n_cohort), 4))
        cov = cov.orderBy(F.desc("measurements"))
        cov.show(truncate=False)
        run.param("coverage", [r.asDict() for r in cov.collect()])  # DRIVER-SIDE ONLY

        aggs = [F.first(a).alias(a) for a in C.CHART_AGGS]
        wide = long.groupBy("stay_id").pivot("concept", CONCEPTS).agg(*aggs)

        feats = cohort.join(wide, "stay_id", "left")
        for c in CONCEPTS:
            cnt = c + "_count"
            miss = F.when(F.col(cnt).isNull() | (F.col(cnt) == F.lit(0)), F.lit(1)).otherwise(F.lit(0))
            feats = feats.withColumn(c + "_missing", miss)
            feats = feats.withColumn(cnt, F.coalesce(F.col(cnt), F.lit(0)))

        n_out = run.count("feature_rows", feats.count())
        if n_out != n_cohort:
            raise ValueError("left join changed row count: %d -> %d" % (n_cohort, n_out))

        run.count("n_feature_columns", len(feats.columns))
        print("feature columns:", feats.columns)

        out = C.PATH_FEATURES
        if args.fraction < 1.0:
            out = "%s_frac%d" % (C.PATH_FEATURES, int(args.fraction * 100))
        feats.coalesce(8).write.mode("overwrite").parquet(out)
        run.output(out)

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
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--shuffle", type=int, default=200)
    main(p.parse_args())