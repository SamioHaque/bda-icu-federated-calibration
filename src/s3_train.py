"""S3: train LogisticRegression and GBTClassifier across data-governance regimes.

Regimes
  R1_pooled   -- one model on all training data, ignoring partition structure.
                 Trained ONCE per (seed, model): pooled training is identical
                 under both partition conditions, so refitting per condition
                 would be wasted work and would invite seed-level disagreement
                 between two runs that ought to be the same model.
  R2_local    -- one model per client, trained on that client's rows only.
                 Each client model is scored on the FULL pooled evaluation set,
                 so the (train_client, eval_client) grid gives the cross-client
                 transfer matrix for free: the diagonal is local performance,
                 off-diagonal is performance under distribution shift.
  R2e_ensemble-- unweighted mean of the R2_local predicted probabilities. This
                 is a one-shot posterior aggregation baseline, NOT federated
                 learning: no iterative averaging, no privacy mechanism.

This script emits PREDICTIONS ONLY; all metrics are computed in S4. Separating
them means a new metric or calibration method never requires refitting.

Predictions are emitted for BOTH the validation and test splits. S4 fits
post-hoc calibration on validation and applies it to test; scoring the test
split before calibration is fitted would leak the test set into calibration.

No class weighting. Rebalancing would push predicted probabilities away from
observed frequencies by construction, which would make miscalibration an
artifact of the training recipe rather than a finding about the regime. Test
prevalence stays natural at ~10.8%.
"""
import argparse
import time

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.functions import vector_to_array

import config as C
from manifest import Run

LABEL = "hospital_expire_flag"
CONCEPTS = list(C.CHART_CONCEPTS.keys())

# Aggregates that can be null (no measurement, or std with a single reading)
# and therefore need median imputation fitted on TRAIN ONLY.
IMPUTE_COLS = ["%s_%s" % (c, a) for c in CONCEPTS for a in C.CHART_AGGS if a != "count"]
IMPUTED_OUT = [c + "_i" for c in IMPUTE_COLS]
COUNT_COLS = ["%s_count" % c for c in CONCEPTS]
MISS_COLS = ["%s_missing" % c for c in CONCEPTS]
EXTRA_NUM = ["anchor_age"]

ASSEMBLE = IMPUTED_OUT + COUNT_COLS + MISS_COLS + EXTRA_NUM + ["gender_idx"]

PRED_COLS = ["seed", "condition", "model", "regime", "train_client",
             "eval_client", "split", "stay_id", "label", "p"]


def load(spark, seed):
    """Feature matrix joined to splits. first_careunit is NEVER joined in:
    it is the partition key for the non-IID condition, so including it as a
    feature would leak the partition into the model."""
    feats = spark.read.parquet(C.PATH_FEATURES).drop("intime")
    demo = spark.read.parquet(C.PATH_COHORT).select("stay_id", "anchor_age", "gender")
    sp = (spark.read.parquet(C.PATH_SPLITS)
          .filter(F.col("seed") == seed)
          .select("stay_id", "split", "client_iid", "client_careunit"))

    df = feats.join(demo, "stay_id").join(sp, "stay_id")

    # MLlib requires double for Imputer/VectorAssembler inputs.
    for c in IMPUTE_COLS + COUNT_COLS + MISS_COLS + EXTRA_NUM:
        df = df.withColumn(c, F.col(c).cast("double"))
    return df.withColumn(LABEL, F.col(LABEL).cast("double"))


def build_pipeline(model_name, seed):
    imp = Imputer(inputCols=IMPUTE_COLS, outputCols=IMPUTED_OUT, strategy="median")
    idx = StringIndexer(inputCol="gender", outputCol="gender_idx", handleInvalid="keep")
    asm = VectorAssembler(inputCols=ASSEMBLE, outputCol="features_raw", handleInvalid="keep")
    sca = StandardScaler(inputCol="features_raw", outputCol="features",
                         withMean=True, withStd=True)
    if model_name == "lr":
        clf = LogisticRegression(featuresCol="features", labelCol=LABEL,
                                 maxIter=100, regParam=0.01, elasticNetParam=0.0)
    elif model_name == "gbt":
        clf = GBTClassifier(featuresCol="features", labelCol=LABEL,
                            maxIter=30, maxDepth=5, stepSize=0.1, seed=seed)
    else:
        raise ValueError("unknown model: %s" % model_name)
    return Pipeline(stages=[imp, idx, asm, sca, clf])


def score(model, holdout, seed, condition, model_name, regime, train_client):
    """Predict on the holdout (val + test) and shape into the flat schema.
    eval_client is the row's own client under the active condition, so the
    per-client breakdown and the transfer matrix both fall out of a groupBy."""
    out = model.transform(holdout)
    out = out.withColumn("p", vector_to_array("probability")[1])
    return out.select(
        F.lit(seed).cast("int").alias("seed"),
        F.lit(condition).alias("condition"),
        F.lit(model_name).alias("model"),
        F.lit(regime).alias("regime"),
        F.lit(train_client).alias("train_client"),
        F.col("client_" + condition).alias("eval_client"),
        F.col("split"),
        F.col("stay_id"),
        F.col(LABEL).alias("label"),
        F.col("p"),
    )


def main(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    spark = SparkSession.builder.appName("s3_train").getOrCreate()
    spark.conf.set("temporaryGcsBucket", C.BUCKET)
    spark.conf.set("spark.sql.shuffle.partitions", str(args.shuffle))

    run = Run(spark, "s3_train", args.git_sha,
              {"seeds": seeds, "models": models, "conditions": conditions,
               "shuffle_partitions": args.shuffle,
               "n_features": len(ASSEMBLE),
               "class_weighting": "none",
               "lr": {"maxIter": 100, "regParam": 0.01, "elasticNet": 0.0},
               "gbt": {"maxIter": 30, "maxDepth": 5, "stepSize": 0.1}})
    run.await_executors(args.expect_executors)

    fit_times = []
    frames = []
    out = C.PATH_MODELS + "/predictions"
    first_write = True

    try:
        for seed in seeds:
            df = load(spark, seed).cache()
            n_all = run.count("rows_seed%d" % seed, df.count())

            train_all = df.filter(F.col("split") == "train").cache()
            holdout = df.filter(F.col("split").isin("val", "test")).cache()
            n_train = run.count("train_seed%d" % seed, train_all.count())
            run.count("holdout_seed%d" % seed, holdout.count())

            for model_name in models:
                # ---- R1: pooled -------------------------------------------
                t0 = time.time()
                m_pooled = build_pipeline(model_name, seed).fit(train_all)
                dt = round(time.time() - t0, 2)
                fit_times.append({"seed": seed, "model": model_name,
                                  "regime": "R1_pooled", "client": "pooled",
                                  "n_train": n_train, "fit_s": dt})
                print("[fit] seed=%d %s R1_pooled  %.2fs" % (seed, model_name, dt))

                for cond in conditions:
                    frames.append(score(m_pooled, holdout, seed, cond,
                                        model_name, "R1_pooled", "pooled"))

                if args.save_models and seed == seeds[0]:
                    uri = "%s/%s_R1_pooled_seed%d" % (C.PATH_MODELS, model_name, seed)
                    m_pooled.write().overwrite().save(uri)
                    run.output(uri)

                # ---- R2 / R2e: per client ---------------------------------
                for cond in conditions:
                    ccol = "client_" + cond
                    clients = sorted([r[0] for r in
                                      train_all.select(ccol).distinct().collect()])
                    local_preds = []

                    for cl in clients:
                        sub = train_all.filter(F.col(ccol) == cl)
                        n_sub = sub.count()
                        t0 = time.time()
                        m_local = build_pipeline(model_name, seed).fit(sub)
                        dt = round(time.time() - t0, 2)
                        fit_times.append({"seed": seed, "model": model_name,
                                          "regime": "R2_local", "client": cl,
                                          "n_train": n_sub, "fit_s": dt})
                        print("[fit] seed=%d %s R2_local %s=%s n=%d  %.2fs"
                              % (seed, model_name, cond, cl, n_sub, dt))

                        p = score(m_local, holdout, seed, cond, model_name,
                                  "R2_local", cl)
                        frames.append(p)
                        local_preds.append(p)

                    # R2e: unweighted mean of the local posteriors.
                    ens = local_preds[0]
                    for extra in local_preds[1:]:
                        ens = ens.unionByName(extra)
                    ens = (ens.groupBy("seed", "condition", "model",
                                       "eval_client", "split", "stay_id", "label")
                              .agg(F.avg("p").alias("p"))
                              .withColumn("regime", F.lit("R2e_ensemble"))
                              .withColumn("train_client", F.lit("ensemble"))
                              .select(*PRED_COLS))
                    frames.append(ens)

                # Flush this (seed, model) chunk. Accumulating every frame and
                # unioning once at the end inflates the driver-side query plan;
                # at 3 seeds x 2 models the serialized task binary grows large
                # enough to destabilise the final write.
                chunk = frames[0].select(*PRED_COLS)
                for extra in frames[1:]:
                    chunk = chunk.unionByName(extra.select(*PRED_COLS))
                (chunk.write
                      .mode("overwrite" if first_write else "append")
                      .partitionBy("seed", "condition")
                      .parquet(out))
                print("[write] flushed seed=%d model=%s (%d frames)"
                      % (seed, model_name, len(frames)))
                first_write = False
                frames = []

            df.unpersist()
            train_all.unpersist()
            holdout.unpersist()

        run.param("fit_times", fit_times)  # DRIVER-SIDE ONLY -- small table
        run.param("total_fit_s", round(sum(f["fit_s"] for f in fit_times), 2))
        run.param("n_fits", len(fit_times))
        run.output(out)

        written = spark.read.parquet(out)
        run.count("prediction_rows", written.count())
        written.groupBy("regime", "model", "split").count().orderBy(
            "regime", "model", "split").show(50, truncate=False)

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
    p.add_argument("--seeds", default="42")
    p.add_argument("--models", default="lr,gbt")
    p.add_argument("--conditions", default="iid,careunit")
    p.add_argument("--shuffle", type=int, default=64)
    p.add_argument("--save-models", action="store_true")
    main(p.parse_args())