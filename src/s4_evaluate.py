"""S4: post-hoc calibration and metric computation.

Phase A -- calibration, at two distinct SCOPES

  Every calibrator is fitted on VALIDATION rows only; fitting on test would
  leak the evaluation set. The scope is which validation rows are used:

    raw         no calibration.
    iso_pooled  fitted on the whole pooled validation set for that model.
                Scenario: a central body receives a donated model and
                recalibrates it against the population it will serve.
    iso_target  fitted on the EVALUATION client's own validation rows and
                applied only to that client. Scenario: the receiving site
                recalibrates locally on data it already holds.

  The comparison iso_pooled vs iso_target answers the question that motivates
  the whole calibration analysis: when a model is deployed under prevalence
  shift, does recalibrating at the destination repair it, and by how much?

  NOTE: S3 scores every client model on the full pooled holdout, so grouping by
  train_client and filtering to val yields POOLED validation rows -- not the
  training client's own. Earlier revisions of this file mislabelled iso_pooled
  as source-client calibration; the scope names above are now literal.

Phase B -- metrics
  All metrics are computed in a small number of distributed passes rather than
  by looping an Evaluator over groups, which would launch hundreds of Spark
  jobs. Specifically:

  AUROC   Mann-Whitney rank identity, using average ranks so ties are handled
          correctly:  (sum of positive ranks - n_pos(n_pos+1)/2) / (n_pos*n_neg)
  AUPRC   Average precision:  (1/n_pos) * sum of precision@i over positives,
          with precision@i from a descending cumulative window.
  Brier   mean((p - y)^2)
  ECE/MCE 10 equal-width reliability bins; ECE is the count-weighted mean gap,
          MCE the largest single-bin gap.
  P/R/F1  at a threshold chosen on VALIDATION to maximise F1, then applied
          unchanged to test. Selecting the threshold on test would inflate it.

  eval_client="ALL" rows are added so pooled and per-client performance come
  from one table.
"""
import argparse

from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.regression import IsotonicRegression
from pyspark.ml.functions import array_to_vector

import config as C
from manifest import Run

# Identifies one evaluated model/dataset combination.
GKEY = ["seed", "condition", "model", "regime", "train_client",
        "eval_client", "calibration"]
GROUP = GKEY + ["split"]

PATH_CAL = C.PATH_MODELS + "/predictions_cal"
N_BINS = 10

# Columns present in both prediction tables before calibration is applied.
BASE = ["seed", "condition", "model", "regime", "train_client",
        "eval_client", "split"]

# S3 (MLlib: R1/R2/R2e) and S3b (federated: R3_*) write the same schema, so
# calibration and metric computation are shared across both.
PRED_SOURCES = [C.PATH_MODELS + "/predictions",
                C.PATH_MODELS + "/predictions_fl"]


def read_predictions(spark, run):
    frames, found = [], []
    for p in PRED_SOURCES:
        try:
            f = spark.read.parquet(p).select(*BASE, "stay_id", "label", "p")
            frames.append(f)
            found.append(p)
            print("[read] %s" % p)
        except Exception as e:
            print("[read] SKIPPED %s -- %s" % (p, str(e)[:140]))
    if not frames:
        raise ValueError("no prediction tables found under %s" % C.PATH_MODELS)
    run.param("prediction_sources", found)
    out = frames[0]
    for f in frames[1:]:
        out = out.unionByName(f)
    return out


def _iso():
    return IsotonicRegression(featuresCol="pv", labelCol="label",
                              predictionCol="p_iso", isotonic=True)


def _shape(df, label):
    return (df.withColumn("calibration", F.lit(label))
              .withColumn("p", F.col("p_iso"))
              .select(*GROUP, "stay_id", "label", "p"))


def calibrate(spark, run, args):
    """Emit raw + iso_pooled (+ iso_target) per model group."""
    preds = read_predictions(spark, run)
    keys = [r.asDict() for r in
            preds.select("seed", "condition", "model", "regime", "train_client")
                 .distinct().orderBy("seed", "condition", "model",
                                     "regime", "train_client").collect()]
    print("[cal] %d model groups; target_scope=%s"
          % (len(keys), not args.no_target_scope))

    first = True
    n_fit = 0
    for gi, k in enumerate(keys, 1):
        cond = ((F.col("seed") == k["seed"]) &
                (F.col("condition") == k["condition"]) &
                (F.col("model") == k["model"]) &
                (F.col("regime") == k["regime"]) &
                (F.col("train_client") == k["train_client"]))
        sub = preds.filter(cond).withColumn("pv", array_to_vector(F.array("p")))
        sub.cache()

        parts = [sub.select(*GROUP, "stay_id", "label", "p")
                    .withColumn("calibration", F.lit("raw"))
                    .select(*GROUP, "stay_id", "label", "p")]

        # --- scope 1: pooled target population
        m = _iso().fit(sub.filter(F.col("split") == "val"))
        n_fit += 1
        parts.append(_shape(m.transform(sub), "iso_pooled"))

        # --- scope 2: each evaluation client recalibrates on its own data
        if not args.no_target_scope:
            evs = sorted([r[0] for r in
                          sub.select("eval_client").distinct().collect()])
            for e in evs:
                se = sub.filter(F.col("eval_client") == e)
                ve = se.filter(F.col("split") == "val")
                try:
                    me = _iso().fit(ve)
                except Exception as ex:
                    print("[cal] skip target %s/%s: %s"
                          % (k["train_client"], e, str(ex)[:80]))
                    continue
                n_fit += 1
                parts.append(_shape(me.transform(se), "iso_target"))

        out = parts[0]
        for p in parts[1:]:
            out = out.unionByName(p)
        (out.write.mode("overwrite" if first else "append")
            .partitionBy("seed", "condition").parquet(PATH_CAL))
        first = False
        sub.unpersist()

        if gi % 10 == 0:
            print("[cal] group %d/%d, %d calibrators fitted" % (gi, len(keys), n_fit))

    run.param("n_model_groups", len(keys))
    run.count("calibrators_fitted", n_fit)
    return PATH_CAL


def add_pooled_eval(df):
    """Duplicate every row with eval_client='ALL' so pooled and per-client
    metrics fall out of the same aggregation."""
    return df.unionByName(df.withColumn("eval_client", F.lit("ALL")))


def compute_metrics(spark, run):
    df = spark.read.parquet(PATH_CAL)
    df = add_pooled_eval(df).cache()
    run.count("scored_rows", df.count())

    w_grp = Window.partitionBy(*GROUP)
    w_asc = Window.partitionBy(*GROUP).orderBy(F.col("p").asc())
    w_tie = Window.partitionBy(*(GROUP + ["p"]))
    w_desc = (Window.partitionBy(*GROUP).orderBy(F.col("p").desc())
              .rowsBetween(Window.unboundedPreceding, Window.currentRow))

    d = df.withColumn("n_pos_g", F.sum("label").over(w_grp))
    d = d.withColumn("rn", F.row_number().over(w_asc))
    d = d.withColumn("rank_avg", F.avg("rn").over(w_tie))
    d = d.withColumn("cum_tp", F.sum("label").over(w_desc))
    d = d.withColumn("cum_n", F.count(F.lit(1)).over(w_desc))
    d = d.withColumn("prec_at", F.col("cum_tp") / F.col("cum_n"))
    # F1 at every distinct cut point: TP=cum_tp, FP=cum_n-cum_tp, FN=n_pos-cum_tp
    d = d.withColumn("f1_at",
                     (2.0 * F.col("cum_tp")) /
                     (F.col("cum_n") + F.col("n_pos_g")))
    d = d.cache()

    base = d.groupBy(*GROUP).agg(
        F.count(F.lit(1)).alias("n"),
        F.sum("label").alias("n_pos"),
        F.sum(F.when(F.col("label") == 1, F.col("rank_avg"))
               .otherwise(F.lit(0.0))).alias("sum_rank_pos"),
        F.sum(F.when(F.col("label") == 1, F.col("prec_at"))
               .otherwise(F.lit(0.0))).alias("sum_prec_pos"),
        F.avg(F.pow(F.col("p") - F.col("label"), F.lit(2))).alias("brier"),
        F.avg("p").alias("mean_p"),
    )
    base = base.withColumn("n_neg", F.col("n") - F.col("n_pos"))
    base = base.withColumn("prevalence", F.col("n_pos") / F.col("n"))
    base = base.withColumn(
        "auroc",
        (F.col("sum_rank_pos") - F.col("n_pos") * (F.col("n_pos") + 1) / 2.0) /
        (F.col("n_pos") * F.col("n_neg")))
    base = base.withColumn("auprc", F.col("sum_prec_pos") / F.col("n_pos"))
    base = base.drop("sum_rank_pos", "sum_prec_pos")

    # --- Reliability bins ---------------------------------------------------
    b = df.withColumn("bin", F.least(F.floor(F.col("p") * N_BINS),
                                     F.lit(N_BINS - 1)).cast("int"))
    bins = b.groupBy(*GROUP, "bin").agg(
        F.count(F.lit(1)).alias("bin_n"),
        F.avg("p").alias("bin_conf"),
        F.avg("label").alias("bin_freq"))
    bins = bins.withColumn("gap", F.abs(F.col("bin_conf") - F.col("bin_freq")))

    cal = bins.groupBy(*GROUP).agg(
        (F.sum(F.col("gap") * F.col("bin_n")) / F.sum("bin_n")).alias("ece"),
        F.max("gap").alias("mce"))

    # --- Threshold chosen on validation, applied to test --------------------
    thr = (d.filter(F.col("split") == "val")
            .groupBy(*GKEY)
            .agg(F.max(F.struct(F.col("f1_at"), F.col("p"))).alias("best")))
    thr = thr.select(*GKEY,
                     F.col("best.p").alias("threshold"),
                     F.col("best.f1_at").alias("val_f1"))

    scored = d.join(thr, GKEY, "left")
    scored = scored.withColumn("yhat",
                               (F.col("p") >= F.col("threshold")).cast("double"))
    pr = scored.groupBy(*GROUP).agg(
        F.first("threshold").alias("threshold"),
        F.sum(F.when((F.col("yhat") == 1) & (F.col("label") == 1), 1)
               .otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("yhat") == 1) & (F.col("label") == 0), 1)
               .otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("yhat") == 0) & (F.col("label") == 1), 1)
               .otherwise(0)).alias("fn"))
    pr = pr.withColumn("precision",
                       F.col("tp") / F.greatest(F.col("tp") + F.col("fp"), F.lit(1)))
    pr = pr.withColumn("recall",
                       F.col("tp") / F.greatest(F.col("tp") + F.col("fn"), F.lit(1)))
    pr = pr.withColumn("f1",
                       (2.0 * F.col("precision") * F.col("recall")) /
                       F.greatest(F.col("precision") + F.col("recall"), F.lit(1e-12)))

    metrics = base.join(cal, GROUP).join(pr, GROUP)
    metrics = metrics.orderBy(*GROUP)

    return metrics, bins


def main(args):
    spark = SparkSession.builder.appName("s4_evaluate").getOrCreate()
    spark.conf.set("temporaryGcsBucket", C.BUCKET)
    spark.conf.set("spark.sql.shuffle.partitions", str(args.shuffle))

    run = Run(spark, "s4_evaluate", args.git_sha,
              {"n_bins": N_BINS,
               "calibration": "isotonic",
               "calibration_scopes": ["raw", "iso_pooled"] +
                                     ([] if args.no_target_scope else ["iso_target"]),
               "iso_pooled_fitted_on": "pooled validation split (all eval clients)",
               "iso_target_fitted_on": "evaluation client's own validation split",
               "threshold_selected_on": "validation, max F1",
               "shuffle_partitions": args.shuffle,
               "skip_calibration": args.skip_calibration})
    run.await_executors(args.expect_executors)

    try:
        if not args.skip_calibration:
            calibrate(spark, run, args)
        else:
            print("[cal] skipped; reusing %s" % PATH_CAL)

        metrics, bins = compute_metrics(spark, run)

        m_uri = C.PATH_METRICS + "/metrics"
        b_uri = C.PATH_METRICS + "/reliability"
        metrics.coalesce(1).write.mode("overwrite").option("header", True).csv(m_uri)
        bins.coalesce(1).write.mode("overwrite").option("header", True).csv(b_uri)
        run.output(m_uri)
        run.output(b_uri)
        run.output(PATH_CAL)
        run.count("metric_rows", spark.read.option("header", True).csv(m_uri).count())

        # Headline view: pooled test performance, both calibrations.
        head = (metrics
                .filter((F.col("split") == "test") & (F.col("eval_client") == "ALL"))
                .select("seed", "condition", "model", "regime", "train_client",
                        "calibration", "n", "prevalence",
                        F.round("auroc", 4).alias("auroc"),
                        F.round("auprc", 4).alias("auprc"),
                        F.round("brier", 5).alias("brier"),
                        F.round("ece", 5).alias("ece"),
                        F.round("f1", 4).alias("f1"))
                .orderBy("model", "regime", "train_client", "condition",
                         "calibration", "seed"))
        head.show(200, truncate=False)

        print("=== coverage: regimes x models present in metrics ===")
        (metrics.filter(F.col("split") == "test")
                .groupBy("model", "regime", "calibration")
                .agg(F.countDistinct("seed").alias("seeds"),
                     F.count(F.lit(1)).alias("cells"))
                .orderBy("model", "regime", "calibration")
                .show(60, truncate=False))

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
    p.add_argument("--shuffle", type=int, default=64)
    p.add_argument("--skip-calibration", action="store_true",
                   help="reuse an existing predictions_cal; recompute metrics only")
    p.add_argument("--no-target-scope", action="store_true",
                   help="skip per-eval-client recalibration (faster, fewer fits)")
    main(p.parse_args())