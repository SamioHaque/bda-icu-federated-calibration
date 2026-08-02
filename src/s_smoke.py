"""Phase 0 smoke test. Verifies BQ connector read, GCS write, manifest logging."""
import argparse
from pyspark.sql import SparkSession, functions as F

import config as C
from manifest import Run


def main(args):
    spark = SparkSession.builder.appName("s_smoke").getOrCreate()
    spark.conf.set("temporaryGcsBucket", C.BUCKET)
    run = Run(spark, "s_smoke", args.git_sha, {"seed": C.SEED})
    run.await_executors(4)

    try:
        icu = (spark.read.format("bigquery")
               .option("table", f"{C.MIMIC_ICU}.icustays")
               .option("parentProject", C.PROJECT_ID)
               .load())
        icu.printSchema()
        run.count("icustays_rows", icu.count())
        run.count("icustays_subjects", icu.select("subject_id").distinct().count())

        by_unit = (icu.groupBy("first_careunit")
                     .agg(F.count("*").alias("n"))
                     .orderBy(F.desc("n")))
        by_unit.show(20, truncate=False)

        uri = f"{C.GCS_ROOT}/_smoke/careunit_counts"
        by_unit.coalesce(1).write.mode("overwrite").parquet(uri)
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
    main(p.parse_args())