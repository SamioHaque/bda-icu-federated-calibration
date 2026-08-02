"""S3b: federated logistic regression with multiple server-side aggregation strategies.

Implements genuine multi-round federated optimisation on Spark: FedAvg
(McMahan et al., 2017), FedProx (Li et al., 2020), and the adaptive server
optimisers FedAdagrad / FedAdam / FedYogi (Reddi et al., 2021).

Why this is not implementable with pyspark.ml estimators
-------------------------------------------------------
MLlib's LogisticRegression exposes no warm-start: a fit cannot be initialised
from supplied coefficients. Multi-round FL requires exactly that. So the local
solver is implemented directly, with the gradient computed by a distributed
treeReduce over each client's partition of the cached RDD -- the same
map/reduce structure MLlib uses internally. Everything stays in Spark; nothing
is pulled to the driver except (d+3)-sized gradient accumulators and, at
evaluation rounds, a few thousand probabilities.

Federated preprocessing
-----------------------
Imputation and standardisation constants are derived from per-client
SUFFICIENT STATISTICS ONLY (count, sum, sum-of-squares per feature), combined
on the server by a size-weighted average. No raw record leaves its client
partition at any point, including during preprocessing. Median imputation is
deliberately replaced by mean imputation because the mean is exactly
federatable from sufficient statistics and the median is not.

Honest scope
------------
Clients are simulated partitions of one dataset inside one Spark cluster (as in
Vieira et al.'s Flower setup). Parameters are exchanged rather than records,
but NO differential privacy and NO secure aggregation are implemented, so this
supports no formal privacy guarantee. GBT is absent by necessity: decision
trees cannot be parameter-averaged.

The R3_central arm runs this identical code path with K=1, giving an
apples-to-apples centralised ceiling that shares the preprocessing, optimiser
family, and numerics of the federated arms. MLlib's R1_pooled remains an
independent anchor.
"""
import argparse
import math
import time

import numpy as np
from pyspark.sql import SparkSession, functions as F

import config as C
from manifest import Run

LABEL = "hospital_expire_flag"
CONCEPTS = list(C.CHART_CONCEPTS.keys())

AGG_NUM = ["%s_%s" % (c, a) for c in CONCEPTS for a in C.CHART_AGGS]
MISS = ["%s_missing" % c for c in CONCEPTS]
NUMERIC = AGG_NUM + ["anchor_age"]          # mean-imputed then standardised
FEATURES = NUMERIC + MISS + ["gender_m"]    # fixed column order
D = len(FEATURES)

ADAPTIVE = ("fedadagrad", "fedadam", "fedyogi")
PRED_COLS = ["seed", "condition", "model", "regime", "train_client",
             "eval_client", "split", "stay_id", "label", "p"]


# ---------------------------------------------------------------- data loading

def load(spark, seed):
    feats = spark.read.parquet(C.PATH_FEATURES).drop("intime")
    demo = spark.read.parquet(C.PATH_COHORT).select("stay_id", "anchor_age", "gender")
    sp = (spark.read.parquet(C.PATH_SPLITS)
          .filter(F.col("seed") == seed)
          .select("stay_id", "split", "client_iid", "client_careunit"))
    df = feats.join(demo, "stay_id").join(sp, "stay_id")
    for c in NUMERIC + MISS:
        df = df.withColumn(c, F.col(c).cast("double"))
    return df.withColumn(LABEL, F.col(LABEL).cast("double"))


def federated_stats(train, ccol):
    """Per-client sufficient statistics -> size-weighted global mean/std.

    Returns {col: (mean, std)} plus the per-client statistic table for the
    manifest. Only aggregates cross the client boundary.
    """
    aggs = []
    for c in NUMERIC:
        aggs += [F.count(c).alias("n__" + c),
                 F.sum(c).alias("s__" + c),
                 F.sum(F.col(c) * F.col(c)).alias("q__" + c)]
    # DRIVER-SIDE ONLY -- K rows of aggregates, no record-level data
    rows = train.groupBy(ccol).agg(*aggs).collect()

    out = {}
    for c in NUMERIC:
        n = sum(r["n__" + c] or 0 for r in rows)
        s = sum(r["s__" + c] or 0.0 for r in rows)
        q = sum(r["q__" + c] or 0.0 for r in rows)
        if n == 0:
            out[c] = (0.0, 1.0)
            continue
        mean = s / n
        var = max(q / n - mean * mean, 0.0)
        out[c] = (mean, math.sqrt(var) if var > 1e-12 else 1.0)
    return out, len(rows)


def feature_exprs(stats):
    e = []
    for c in NUMERIC:
        m, s = stats[c]
        e.append(((F.coalesce(F.col(c), F.lit(m)) - F.lit(m)) / F.lit(s)).alias(c))
    for c in MISS:
        e.append(F.col(c).alias(c))
    e.append(F.when(F.col("gender") == "M", F.lit(1.0)).otherwise(F.lit(0.0))
             .alias("gender_m"))
    return e


def to_rdd(df, stats, client_map, ccol, n_parts):
    """Cached RDD of (client_idx, label, np.array(D))."""
    cols = feature_exprs(stats)
    idx = F.create_map([F.lit(x) for kv in client_map.items() for x in kv])
    sel = df.select(idx[F.col(ccol)].alias("k"), F.col(LABEL).alias("y"),
                    F.array(*cols).alias("x"))
    rdd = (sel.rdd
           .map(lambda r: (int(r["k"]), float(r["y"]),
                           np.asarray(r["x"], dtype=np.float64)))
           .repartition(n_parts)
           .cache())
    rdd.count()
    return rdd


# ------------------------------------------------------------ distributed math

def _sigmoid(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def grad_pass(rdd, sc, W, K):
    """One distributed pass. Returns (K, D+3) accumulator:
    [:D] coef gradient sum, [D] bias gradient sum, [D+1] n, [D+2] logloss sum.

    All K client gradients are computed in a SINGLE pass by looking up each
    row's own client weights, so cost is O(rounds x local_epochs), not
    O(rounds x local_epochs x clients)."""
    bw = sc.broadcast(W)
    width = D + 3

    def part(it):
        acc = np.zeros((K, width))
        Wl = bw.value
        for k, y, x in it:
            w = Wl[k]
            z = float(np.dot(w[:D], x)) + w[D]
            p = _sigmoid(z)
            acc[k, :D] += (p - y) * x
            acc[k, D] += (p - y)
            acc[k, D + 1] += 1.0
            pc = min(max(p, 1e-12), 1.0 - 1e-12)
            acc[k, D + 2] -= y * math.log(pc) + (1.0 - y) * math.log(1.0 - pc)
        yield acc

    out = rdd.mapPartitions(part).treeReduce(lambda a, b: a + b, depth=2)
    bw.destroy()
    return out


def collect_probs(rdd, sc, w):
    """DRIVER-SIDE ONLY -- a few thousand (p, y) pairs for a convergence check."""
    bw = sc.broadcast(w)

    def part(it):
        wl = bw.value
        for k, y, x in it:
            z = float(np.dot(wl[:D], x)) + wl[D]
            yield (_sigmoid(z), y)

    out = rdd.mapPartitions(part).collect()
    bw.destroy()
    return out


def auroc(pairs):
    """Mann-Whitney with average ranks."""
    pairs = sorted(pairs, key=lambda t: t[0])
    n = len(pairs)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    y = np.array([p[1] for p in pairs])
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


# ------------------------------------------------------------------- FL driver

def server_update(strategy, w, delta, st, a):
    if strategy in ("fedavg", "fedprox", "central"):
        return w + a.server_lr * delta
    if strategy == "fedadagrad":
        st["v"] += delta * delta
        return w + a.server_lr * delta / (np.sqrt(st["v"]) + a.tau)
    st["m"] = a.beta1 * st["m"] + (1.0 - a.beta1) * delta
    if strategy == "fedadam":
        st["v"] = a.beta2 * st["v"] + (1.0 - a.beta2) * delta * delta
    else:  # fedyogi
        d2 = delta * delta
        st["v"] = st["v"] - (1.0 - a.beta2) * d2 * np.sign(st["v"] - d2)
    return w + a.server_lr * st["m"] / (np.sqrt(st["v"]) + a.tau)


def run_strategy(sc, strategy, train_rdd, val_rdd, sizes, K, a, tag):
    """Multi-round federated training. Returns (weights, history)."""
    w = np.zeros(D + 1)
    st = {"m": np.zeros(D + 1), "v": np.full(D + 1, a.tau ** 2)}
    total = float(sum(sizes.values()))
    mu = a.mu if strategy == "fedprox" else 0.0
    hist = []
    t0 = time.time()

    for rnd in range(1, a.rounds + 1):
        W = {k: w.copy() for k in range(K)}
        train_loss = None

        for e in range(a.local_epochs):
            G = grad_pass(train_rdd, sc, W, K)
            if e == 0:
                n_all = G[:, D + 1].sum()
                train_loss = float(G[:, D + 2].sum() / max(n_all, 1.0))
            for k in range(K):
                nk = G[k, D + 1]
                if nk <= 0:
                    continue
                g = G[k, :D + 1] / nk
                g[:D] += a.l2 * W[k][:D]
                if mu > 0.0:
                    g += mu * (W[k] - w)
                W[k] = W[k] - a.local_lr * g

        delta = np.zeros(D + 1)
        for k in range(K):
            delta += (sizes[k] / total) * (W[k] - w)
        w = server_update(strategy, w, delta, st, a)

        rec = {"round": rnd, "train_loss": round(train_loss, 6),
               "delta_norm": round(float(np.linalg.norm(delta)), 6),
               "elapsed_s": round(time.time() - t0, 2)}
        if rnd % a.eval_every == 0 or rnd == a.rounds:
            pr = collect_probs(val_rdd, sc, w)
            rec["val_auroc"] = round(auroc(pr), 5)
            rec["val_logloss"] = round(
                float(np.mean([-(y * math.log(max(p, 1e-12)) +
                                 (1 - y) * math.log(max(1 - p, 1e-12)))
                               for p, y in pr])), 6)
            print("[%s] round %3d  train_loss=%.5f  val_auroc=%.4f  (%.1fs)"
                  % (tag, rnd, train_loss, rec["val_auroc"], rec["elapsed_s"]))
        hist.append(rec)

    return w, hist


def predict(df, stats, w, seed, condition, regime, ccol):
    """Score val+test with the final global weights, in the S3 schema."""
    cols = feature_exprs(stats)
    z = F.lit(float(w[D]))
    for j, c in enumerate(cols):
        z = z + F.lit(float(w[j])) * c
    p = F.lit(1.0) / (F.lit(1.0) + F.exp(-z))
    return (df.filter(F.col("split").isin("val", "test"))
              .select(F.lit(seed).cast("int").alias("seed"),
                      F.lit(condition).alias("condition"),
                      F.lit("lr_fl").alias("model"),
                      F.lit(regime).alias("regime"),
                      F.lit("federated").alias("train_client"),
                      F.col(ccol).alias("eval_client"),
                      F.col("split"),
                      F.col("stay_id"),
                      F.col(LABEL).alias("label"),
                      p.alias("p")))


def main(a):
    seeds = [int(s) for s in a.seeds.split(",")]
    conditions = [c.strip() for c in a.conditions.split(",")]
    strategies = [s.strip() for s in a.strategies.split(",")]

    spark = SparkSession.builder.appName("s3b_fedavg").getOrCreate()
    spark.conf.set("temporaryGcsBucket", C.BUCKET)
    spark.conf.set("spark.sql.shuffle.partitions", str(a.shuffle))
    sc = spark.sparkContext

    run = Run(spark, "s3b_fedavg", a.git_sha,
              {"seeds": seeds, "conditions": conditions,
               "strategies": strategies, "rounds": a.rounds,
               "local_epochs": a.local_epochs, "local_lr": a.local_lr,
               "server_lr": a.server_lr, "l2": a.l2, "mu": a.mu,
               "beta1": a.beta1, "beta2": a.beta2, "tau": a.tau,
               "n_features": D, "partitions": a.parts,
               "preprocessing": "mean-impute + standardise from federated "
                                "sufficient statistics (count, sum, sumsq)",
               "privacy": "none -- no DP, no secure aggregation"})
    run.await_executors(a.expect_executors)

    out = C.PATH_MODELS + "/predictions_fl"
    first = True
    history = []
    timing = []

    try:
        for seed in seeds:
            df = load(spark, seed).cache()
            df.count()

            for condition in conditions:
                ccol = "client_" + condition
                train = df.filter(F.col("split") == "train")
                stats, n_clients = federated_stats(train, ccol)

                names = sorted([r[0] for r in
                                train.select(ccol).distinct().collect()])
                cmap = {nm: i for i, nm in enumerate(names)}
                sizes = {cmap[r[0]]: r[1] for r in
                         train.groupBy(ccol).count().collect()}

                tr_rdd = to_rdd(train, stats, cmap, ccol, a.parts)
                val = df.filter(F.col("split") == "val")
                va_rdd = to_rdd(val, stats, cmap, ccol, a.parts)

                # Centralised arm: identical code path, one client.
                tr_one = tr_rdd.map(lambda t: (0, t[1], t[2])).cache()
                tr_one.count()
                va_one = va_rdd.map(lambda t: (0, t[1], t[2])).cache()
                va_one.count()

                for strategy in strategies:
                    if strategy == "central":
                        rdd_t, rdd_v, K = tr_one, va_one, 1
                        sz = {0: float(sum(sizes.values()))}
                    else:
                        rdd_t, rdd_v, K = tr_rdd, va_rdd, len(cmap)
                        sz = sizes

                    tag = "s%d/%s/%s" % (seed, condition, strategy)
                    t0 = time.time()
                    w, hist = run_strategy(sc, strategy, rdd_t, rdd_v,
                                           sz, K, a, tag)
                    dt = round(time.time() - t0, 2)

                    regime = "R3_" + strategy
                    for h in hist:
                        h.update({"seed": seed, "condition": condition,
                                  "strategy": strategy})
                    history.extend(hist)
                    timing.append({"seed": seed, "condition": condition,
                                   "strategy": strategy, "rounds": a.rounds,
                                   "passes": a.rounds * a.local_epochs,
                                   "n_clients": K, "wall_s": dt,
                                   "final_val_auroc": hist[-1].get("val_auroc")})

                    preds = predict(df, stats, w, seed, condition, regime, ccol)
                    (preds.select(*PRED_COLS).write
                          .mode("overwrite" if first else "append")
                          .partitionBy("seed", "condition")
                          .parquet(out))
                    first = False
                    print("[write] %s -> %s (%.1fs)" % (tag, regime, dt))

                for r in (tr_rdd, va_rdd, tr_one, va_one):
                    r.unpersist()
            df.unpersist()

        run.param("history", history)      # DRIVER-SIDE ONLY -- small table
        run.param("timing", timing)
        run.output(out)
        run.count("prediction_rows", spark.read.parquet(out).count())

        h_uri = C.PATH_METRICS + "/fl_history"
        spark.createDataFrame(history).coalesce(1).write.mode("overwrite") \
             .option("header", True).csv(h_uri)
        run.output(h_uri)

        spark.createDataFrame(timing).orderBy("seed", "condition",
                                              "strategy").show(60, False)
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
    p.add_argument("--conditions", default="iid,careunit")
    p.add_argument("--strategies",
                   default="central,fedavg,fedprox,fedadam,fedadagrad")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--local-lr", type=float, default=0.5)
    p.add_argument("--server-lr", type=float, default=1.0)
    p.add_argument("--l2", type=float, default=0.01)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--shuffle", type=int, default=64)
    p.add_argument("--parts", type=int, default=8)
    main(p.parse_args())