"""S5: scalability. Wall-clock vs executor count x data fraction.

Measures two workloads separately, because the expectation is that they behave
in opposite ways:

  --stage s2   The distributed ETL: full chartevents scan -> itemid filter ->
               broadcast 24h-window join -> group/aggregate -> pivot -> write.
               Work grows with the raw event volume, so this should scale with
               executors.

  --stage s3   The MLlib fit on the pooled training set (39k rows, 58
               features). Training data is cached and materialised BEFORE the
               timer starts, so the measurement isolates the distributed fit
               from I/O. At this data size the per-iteration coordination cost
               dominates, so this is expected NOT to scale -- which is the
               point of reporting both curves.

Executor count is varied per JOB, not per cluster: with dynamicAllocation
disabled, `--properties spark.executor.instances=N` requests N executors from a
cluster that has capacity for more. Running every sweep point inside ONE
cluster session matters -- identical work measured on two freshly created
clusters differed by 28% in earlier runs, so cross-cluster variance would swamp
the effect being measured.

Feature-engineering and model-building logic is imported from s2_features and
s3_train rather than duplicated, so the timings describe the real pipeline.
"""
import argparse
import time

from pyspark.sql import SparkSession, functions as F

import config as C
from manifest import Run
import s2_features as S2
import s3_train as S3

SCRATCH = C.GCS_ROOT + "/_scratch/s5"
OUT = C.PATH_METRICS + "/scalability"


def subsample(df, fraction, key="subject_id", salt="frac"):
    """Hash-based subsample on subject_id: whole patients in or out, and
    identical membership for a given fraction across every run."""
    if fraction >= 1.0:
        return df
    keep = int(round(fraction * 100))
    h = F.abs(F.hash(F.concat_ws("|", F.lit(salt), F.col(key)))) % 100
    return df.filter(h < F.lit(keep))


def workload_s2(spark, run, args):
    """Timing replica of the S2 feature build."""
    cohort = spark.read.parquet(C.PATH_COHORT).select(
        "subject_id", "stay_id", "intime", "hospital_expire_flag")
    cohort = subsample(cohort, args.fraction)
    n_cohort = run.count("cohort_rows", cohort.count())

    win = cohort.select("stay_id", "intime")
    win = win.withColumn("t0", F.unix_timestamp("intime"))
    win = win.withColumn("t1", F.col("t0") + F.lit(S2.WINDOW_S))
    win = win.select("stay_id", "t0", "t1")

    ce = (spark.read.format("bigquery")
          .option("table", "%s.%s.chartevents" % (C.PROJECT_ID, C.DATASET))
          .option("parentProject", C.PROJECT_ID)
          .load()
          .select("stay_id", "itemid", "charttime", "valuenum"))

    # ---- timed region begins
    t0 = time.time()

    events = ce.filter(F.col("itemid").isin(S2.ALL_ITEMIDS))
    events = events.filter(F.col("valuenum").isNotNull())
    events = events.join(F.broadcast(win), "stay_id")
    events = events.withColumn("t", F.unix_timestamp("charttime"))
    events = events.filter(F.col("t") >= F.col("t0"))
    events = events.filter(F.col("t") < F.col("t1"))
    events = events.withColumn("concept", S2.concept_col())
    events = events.withColumn("value", S2.value_col())
    events = events.filter(F.col("concept").isNotNull())
    events = events.withColumn("_ok", S2.in_range_col())
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
    long = long.drop("_f", "_l")

    aggs = [F.first(a).alias(a) for a in C.CHART_AGGS]
    wide = long.groupBy("stay_id").pivot("concept", S2.CONCEPTS).agg(*aggs)
    feats = cohort.join(wide, "stay_id", "left")

    uri = "%s/features_e%d_f%d" % (SCRATCH, args.exec_label,
                                   int(args.fraction * 100))
    feats.coalesce(8).write.mode("overwrite").parquet(uri)

    dt = time.time() - t0
    # ---- timed region ends

    run.count("feature_rows", spark.read.parquet(uri).count())
    run.param("scratch_uri", uri)
    return dt, n_cohort


def workload_s3(spark, run, args):
    """Timing replica of the MLlib pooled fit. I/O excluded by design."""
    df = S3.load(spark, C.SEED)
    train = df.filter(F.col("split") == "train")
    train = subsample(train, args.fraction)
    train = train.cache()
    n_train = run.count("train_rows", train.count())   # materialise before timing

    # ---- timed region begins
    t0 = time.time()
    model = S3.build_pipeline(args.model, C.SEED).fit(train)
    dt = time.time() - t0
    # ---- timed region ends

    # Touch the model so the fit cannot be deferred.
    stage = model.stages[-1]
    run.param("model_summary",
              {"n_stages": len(model.stages),
               "clf": type(stage).__name__})
    train.unpersist()
    return dt, n_train


def main(args):
    spark = SparkSession.builder.appName("s5_%s" % args.stage).getOrCreate()
    spark.conf.set("temporaryGcsBucket", C.BUCKET)
    spark.conf.set("spark.sql.shuffle.partitions", str(args.shuffle))

    run = Run(spark, "s5_scalability", args.git_sha,
              {"stage": args.stage, "fraction": args.fraction,
               "model": args.model if args.stage == "s3" else None,
               "requested_executors": args.exec_label,
               "shuffle_partitions": args.shuffle,
               "seed": C.SEED})
    live = run.await_executors(args.expect_executors)

    try:
        if args.stage == "s2":
            dt, n = workload_s2(spark, run, args)
        elif args.stage == "s3":
            dt, n = workload_s3(spark, run, args)
        else:
            raise ValueError("unknown stage: %s" % args.stage)

        run.param("measured_s", round(dt, 2))
        run.param("live_executors_measured", live)
        cores = live * int(spark.conf.get("spark.executor.cores", "2"))
        run.param("total_cores", cores)

        print("[s5] stage=%s fraction=%.2f requested=%d live=%d cores=%d "
              "measured=%.2fs rows=%d"
              % (args.stage, args.fraction, args.exec_label, live, cores, dt, n))

        # One row per sweep point, appended; the curve is built from these.
        row = [{"stage": args.stage,
                "model": args.model if args.stage == "s3" else "na",
                "fraction": float(args.fraction),
                "requested_executors": int(args.exec_label),
                "live_executors": int(live),
                "total_cores": int(cores),
                "shuffle_partitions": int(args.shuffle),
                "n_rows": int(n),
                "measured_s": float(round(dt, 2)),
                "run_id": run.rec["run_id"],
                "git_sha": args.git_sha}]
        spark.createDataFrame(row).coalesce(1).write.mode("append").parquet(OUT)
        run.output(OUT)

        run.finish(C.PATH_MANIFEST, "ok")
    except Exception as e:
        run.finish(C.PATH_MANIFEST, "failed", e)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--git-sha", default="nogit")
    p.add_argument("--stage", required=True, choices=["s2", "s3"])
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--model", default="lr", choices=["lr", "gbt"])
    p.add_argument("--expect-executors", type=int, default=4)
    p.add_argument("--exec-label", type=int, default=4,
                   help="executor count requested via --properties, for the record")
    p.add_argument("--shuffle", type=int, default=200)
    main(p.parse_args())