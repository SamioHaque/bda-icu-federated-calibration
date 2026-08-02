"""S0: cohort construction. Adults, first ICU stay per patient, mortality label.

Emits ONLY columns knowable at ICU hour 24. los, outtime, dod, and
last_careunit are never selected -- leakage is engineered out at the schema
level rather than dropped downstream, so it cannot silently reappear.
"""
import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

import config as C
from manifest import Run

KEEP = ["subject_id", "hadm_id", "stay_id", "intime",
        "anchor_age", "gender", "first_careunit", "hospital_expire_flag"]


def read_bq(spark, table):
    return (spark.read.format("bigquery")
            .option("table", f"{C.PROJECT_ID}.{C.DATASET}.{table}")
            .option("parentProject", C.PROJECT_ID)
            .load())


def main(args):
    spark = SparkSession.builder.appName("s0_cohort").getOrCreate()
    spark.conf.set("temporaryGcsBucket", C.BUCKET)
    run = Run(spark, "s0_cohort", args.git_sha,
              {"seed": C.SEED, "min_age": 18})
    run.await_executors(args.expect_executors)

    try:
        icustays   = read_bq(spark, "icustays")
        patients   = read_bq(spark, "patients")
        admissions = read_bq(spark, "admissions")

        run.count("icustays_raw", icustays.count())

        adult = (icustays
                 .join(patients.select("subject_id", "gender", "anchor_age"),
                       "subject_id")
                 .filter(F.col("anchor_age") >= 18))
        run.count("after_adult_filter", adult.count())

        # Deterministic first stay: stay_id breaks intime ties so the cohort is
        # reproducible across runs regardless of partition order.
        w = Window.partitionBy("subject_id").orderBy(
            F.col("intime").asc(), F.col("stay_id").asc())
        first = (adult.withColumn("_rn", F.row_number().over(w))
                      .filter(F.col("_rn") == 1)
                      .drop("_rn"))
        run.count("after_first_stay", first.count())

        cohort = (first
                  .join(admissions.select("hadm_id", "hospital_expire_flag"),
                        "hadm_id")
                  .filter(F.col("intime").isNotNull())
                  .select(*KEEP)
                  .cache())

        n      = run.count("cohort_rows", cohort.count())
        subj   = run.count("cohort_subjects",
                           cohort.select("subject_id").distinct().count())
        deaths = run.count("cohort_deaths",
                           cohort.agg(F.sum("hospital_expire_flag")).first()[0])
        run.param("prevalence", round(deaths / n, 5))

        if subj != n:
            raise ValueError(f"one row per patient violated: {n} rows, {subj} subjects")

        # --- Non-IID partition diagnostic -------------------------------------
        mapping = F.create_map([F.lit(x) for kv in C.CAREUNIT_GROUPS.items()
                                for x in kv])
        grp = cohort.withColumn("client_group", mapping[F.col("first_careunit")])

        unmapped = grp.filter(F.col("client_group").isNull())
        n_unmapped = unmapped.count()
        if n_unmapped:
            unmapped.select("first_careunit").distinct().show(50, truncate=False)
            raise ValueError(f"{n_unmapped} stays have unmapped first_careunit; "
                             "extend C.CAREUNIT_GROUPS")

        by_grp = (grp.groupBy("client_group")
                     .agg(F.count("*").alias("n"),
                          F.sum("hospital_expire_flag").alias("deaths"))
                     .withColumn("mortality",
                                 F.round(F.col("deaths") / F.col("n"), 4))
                     .orderBy("client_group"))
        by_grp.show(truncate=False)

        # DRIVER-SIDE ONLY -- 3 rows, recorded for the paper's Table 1
        run.param("careunit_groups",
                  [r.asDict() for r in by_grp.collect()])

        # --- Write ------------------------------------------------------------
        cohort.write.mode("overwrite").parquet(C.PATH_COHORT)
        run.output(C.PATH_COHORT)

        # Convenience copy for ad-hoc SQL. Parquet above is canonical; if the
        # connector balks at timestamp_ntz, the pipeline is unaffected.
        try:
            (cohort.write.format("bigquery")
                   .option("table", C.bq("cohort"))
                   .option("parentProject", C.PROJECT_ID)
                   .mode("overwrite").save())
            run.output(f"bq://{C.bq('cohort')}")
        except Exception as e:
            run.param("bq_write_warning", str(e)[:500])
            print(f"[warn] BigQuery convenience write failed: {e}")

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