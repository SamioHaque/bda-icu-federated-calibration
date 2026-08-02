"""S2: first-24h vital aggregates from raw chartevents.

The distributed aggregation over ~433M raw event rows is this project's scale
claim. Features are built from chartevents, never from mimiciv_3_1_derived.

Design notes:
  - Column pruning at read time: 4 of ~15 columns, pushed down to the BQ
    Storage Read API, so the wire volume is a fraction of the table.
  - The cohort (65k rows) is broadcast, so the 24h window join needs no shuffle
    of the large side.
  - first/last use min/max over a struct(charttime, value) rather than a
    window function -- an aggregate, not a sort, so it costs no extra shuffle.
  - Missingness is recorded as an indicator per concept; rows are never dropped
    for incomplete vitals. That choice keeps the cohort denominator fixed at
    S0's 65,366 and avoids the selection bias of a vitals-complete subset.
"""
import argparse
from pyspark.sql import SparkSession, functions as F

import config as C
from manifest import Run

CONCEPTS = list(C.CHART_CONCEPTS.keys())
ALL_ITEMIDS = sorted({i for v in C.CHART_CONCEPTS.values() for i in v})


def concept_col():
    """itemid -> concept name."""
    e = None
    for concept, ids in C.CHART_CONCEPTS.items():
        cond = F.col("itemid").isin(ids)
        e = F.when(cond, F.lit(concept)) if e is None else e.when(cond, F.lit(concept))
    return e.otherwise(F.lit(None))


def value_col():
    """valuenum, with Fahrenheit temperatures converted to Celsius."""
    return F.when(F.col("itemid") == C.TEMP_F_ITEMID,
                  (F.col("valuenum") - 32.0) * 5.0 / 9.0
                  ).otherwise(F.col("valuenum"))


def in_range_col():
    e = None
    for concept, (lo, hi) in C.CHART_RANGES.items():
        cond = (F.col("concept") == concept) & \
               (F.col("value") >= lo) & (F.col("value") <= hi)
        e = F.when(cond, True) if e is None else e.when(cond, True)
    return e.otherwise(False)


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
        cohort = spark.read.parquet(C.PATH_COHORT).select(
            "subject_id", "stay_id", "intime", "hospital_expire_flag")

        # Data-fraction axis for the scalability sweep. Sampling on subject_id
        # keeps whole patients in or out.
        if args.fraction < 1.0:
            keep = int(round(args.fraction * 100))
            cohort = cohort.filter(
                (F.abs(F.hash(F.concat_ws("|", F.lit("frac"), F.col("subject_id")))) % 100) < keep)

        n_cohort = run.count("cohort_rows", cohort.count())

        ce = (spark.read.format("bigquery")
              .option("table", f"{C.PROJECT_ID}.{C.DATASET}.chartevents")
              .option("parentProject", C.PROJECT_ID)
              .load()
              .select("stay_id", "itemid", "charttime", "valuenum"))

        # Metadata-only count: the paper's scale figure.
        run.count("chartevents_rows_total", ce.count())

        events = (ce
                  .filter(F.col("itemid").isin(ALL_ITEMIDS))
                  .filter(F.col("valuenum").isNotNull())
                  .join(F.broadcast(cohort.select("stay_id", "intime")), "stay_id")
                  .filter((F.col("charttime") >= F.col("intime")) &
                          (F.col("charttime") 
                           F.col("intime") + F.expr("INTERVAL 24 HOURS")))
                  .withColumn("concept", concept_col())
                  .withColumn("value", value_col())
                  .filter(F.col("concept").isNotNull()))

        events = events.withColumn("_ok", in_range_col()) \
                       .filter(F.col("_ok")) \
                       .select("stay_id", "concept", "charttime", "value")

        long = (events.groupBy("stay_id", "concept")
                .agg(F.min("value").alias("min"),
                     F.max("value").alias("max"),
                     F.avg("value").alias("mean"),
                     F.stddev_samp("value").alias("std"),
                     F.min(F.struct("charttime", "value")).alias("_f"),
                     F.max(F.struct("charttime", "value")).alias("_l"),
                     F.count("value").alias("count"))
                .withColumn("first", F.col("_f.value"))
                .withColumn("last", F.col("_l.value"))
                .drop("_f", "_l")
                .cache())

        run.count("measurements_used",
                  long.agg(F.sum("count")).first()[0] or 0)
        run.count("stay_concept_pairs", long.count())

        # Per-concept coverage: how many stays have at least one reading.
        cov = (long.groupBy("concept")
                   .agg(F.countDistinct("stay_id").alias("stays"),
                        F.sum("count").alias("measurements"))
                   .withColumn("coverage",
                               F.round(F.col("stays") / F.lit(n_cohort), 4))
                   .orderBy(F.desc("measurements")))
        cov.show(truncate=False)
        run.param("coverage", [r.asDict() for r in cov.collect()])  # DRIVER-SIDE ONLY

        wide = (long.groupBy("stay_id")
                    .pivot("concept", CONCEPTS)
                    .agg(*[F.first(a).alias(a) for a in C.CHART_AGGS]))

        feats = cohort.join(wide, "stay_id", "left")
        for c in CONCEPTS:
            cnt = f"{c}_count"
            feats = (feats
                     .withColumn(f"{c}_missing",
                                 F.when(F.col(cnt).isNull() | (F.col(cnt) == 0), 1)
                                  .otherwise(0))
                     .withColumn(cnt, F.coalesce(F.col(cnt), F.lit(0))))

        n_out = run.count("feature_rows", feats.count())
        if n_out != n_cohort:
            raise ValueError(f"left join changed row count: {n_cohort} -> {n_out}")

        run.count("n_feature_columns", len(feats.columns))
        print("feature columns:", feats.columns)

        out = C.PATH_FEATURES if args.fraction == 1.0 \
            else f"{C.PATH_FEATURES}_frac{int(args.fraction*100)}"
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