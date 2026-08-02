"""S1: deterministic splits and client partition assignment.

S0 guarantees one row per patient, so subject-level and row-level splitting
coincide -- a patient cannot straddle train/test by construction.

Two partition conditions share an identical split assignment, so every regime
and both conditions are evaluated on exactly the same test patients:
  iid       -- hash of subject_id; independent of outcome and care unit
  careunit  -- clinical service grouping (config.CAREUNIT_GROUPS)

first_careunit is consumed here to derive the careunit partition and is NOT
carried into the feature matrix in either condition. Both conditions therefore
use the same features, and the comparison isolates partition structure alone.
"""
import argparse
from pyspark.sql import SparkSession, functions as F

import config as C
from manifest import Run

TRAIN_PCT, VAL_PCT = 60, 80          # <60 train, <80 val, else test
N_CLIENTS = 3


def assign(df, seed):
    """Spark's murmur3 hash is stable across runs and versions, so the split is
    reproducible from (subject_id, seed) alone -- no shuffle, no stored state."""
    sp = F.abs(F.hash(F.concat_ws("|", F.lit(f"split{seed}"),
                                  F.col("subject_id")))) % 100
    cl = F.abs(F.hash(F.concat_ws("|", F.lit(f"client{seed}"),
                                  F.col("subject_id")))) % N_CLIENTS
    mapping = F.create_map([F.lit(x) for kv in C.CAREUNIT_GROUPS.items()
                            for x in kv])
    return (df
            .withColumn("split", F.when(sp < TRAIN_PCT, "train")
                                  .when(sp < VAL_PCT, "val")
                                  .otherwise("test"))
            .withColumn("client_iid", cl.cast("string"))
            .withColumn("client_careunit", mapping[F.col("first_careunit")])
            .withColumn("seed", F.lit(seed)))


def main(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    spark = SparkSession.builder.appName("s1_splits").getOrCreate()
    spark.conf.set("temporaryGcsBucket", C.BUCKET)
    run = Run(spark, "s1_splits", args.git_sha,
              {"seeds": seeds, "train_pct": TRAIN_PCT,
               "val_pct": VAL_PCT - TRAIN_PCT, "test_pct": 100 - VAL_PCT,
               "n_clients": N_CLIENTS})
    run.await_executors(args.expect_executors)

    try:
        cohort = spark.read.parquet(C.PATH_COHORT).select(
            "subject_id", "stay_id", "first_careunit", "hospital_expire_flag")
        n_cohort = run.count("cohort_rows", cohort.count())

        splits = None
        for s in seeds:
            part = assign(cohort, s)
            splits = part if splits is None else splits.unionByName(part)
        splits = splits.select("subject_id", "stay_id", "seed", "split",
                               "client_iid", "client_careunit",
                               "hospital_expire_flag").cache()

        run.count("split_rows", splits.count())

        # --- Invariants: fail loudly rather than produce unexplainable results
        if splits.count() != n_cohort * len(seeds):
            raise ValueError("split row count != cohort x seeds")

        dup = (splits.groupBy("seed", "subject_id").count()
                     .filter(F.col("count") > 1).count())
        if dup:
            raise ValueError(f"{dup} subjects assigned more than once per seed")

        unmapped = splits.filter(F.col("client_careunit").isNull()).count()
        if unmapped:
            raise ValueError(f"{unmapped} rows with unmapped care unit")

        # --- Per-cell diagnostic (long form, both conditions)
        summary = None
        for cond in ("iid", "careunit"):
            s = (splits.groupBy("seed",
                                F.lit(cond).alias("condition"),
                                F.col(f"client_{cond}").alias("client"),
                                "split")
                       .agg(F.count("*").alias("n"),
                            F.sum("hospital_expire_flag").alias("deaths")))
            summary = s if summary is None else summary.unionByName(s)

        summary = (summary
                   .withColumn("prevalence",
                               F.round(F.col("deaths") / F.col("n"), 4))
                   .orderBy("seed", "condition", "client", "split"))

        summary.filter(F.col("seed") == seeds[0]).show(50, truncate=False)

        # Pooled test prevalence per seed -- must stay near cohort-natural 0.1084
        pooled = (splits.filter(F.col("split") == "test")
                        .groupBy("seed")
                        .agg(F.count("*").alias("n"),
                             F.sum("hospital_expire_flag").alias("deaths"))
                        .withColumn("prevalence",
                                    F.round(F.col("deaths") / F.col("n"), 4))
                        .orderBy("seed"))
        pooled.show(truncate=False)

        # DRIVER-SIDE ONLY -- small result tables for the paper
        run.param("pooled_test", [r.asDict() for r in pooled.collect()])
        run.param("cells_seed_%d" % seeds[0],
                  [r.asDict() for r in
                   summary.filter(F.col("seed") == seeds[0]).collect()])

        splits.write.mode("overwrite").partitionBy("seed").parquet(C.PATH_SPLITS)
        run.output(C.PATH_SPLITS)

        uri = f"{C.PATH_METRICS}/split_summary"
        summary.coalesce(1).write.mode("overwrite").option("header", True).csv(uri)
        run.output(uri)

        run.finish(C.PATH_MANIFEST, "ok")
    except Exception as e:
        run.finish(C.PATH_MANIFEST, "failed", e)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--git-sha", default="nogit")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--expect-executors", type=int, default=4)
    main(p.parse_args())