"""S3b: federated training with multiple server-side aggregation strategies.

Implements genuine multi-round federated optimisation on Spark: FedAvg
(McMahan et al., 2017), FedProx (Li et al., 2020), and the adaptive server
optimisers FedAdagrad / FedAdam / FedYogi (Reddi et al., 2021), plus two
single-client controls (`central`, `central_adam`) that isolate the optimiser
from the federation.

TWO MODEL FAMILIES
  --model lr    regularised logistic regression. CONVEX, so FedAvg provably
                reaches the same optimum regardless of client heterogeneity;
                non-IID partitioning degrades the convergence RATE only.
  --model mlp   one hidden layer, tanh activation, sigmoid output.
                NON-CONVEX, so client drift can strand the federation in a
                worse local minimum. This is the arm in which a non-IID
                penalty on FINAL performance can appear at all.

The two share everything except the local gradient and the scoring function.
`server_update`, `run_strategy` and all five strategies operate on a flat
parameter vector and are model-agnostic, which is why adding MLP touched no
aggregation code.

Why the solver is hand-written
------------------------------
pyspark.ml exposes no warm start: a fit cannot be initialised from supplied
coefficients, and every federated round requires exactly that. So the local
solver is implemented directly, with the gradient computed by mapPartitions +
treeReduce over each client's partition of a cached RDD -- the same map/reduce
structure MLlib uses internally. Nothing but a (K, P+2)-sized accumulator
reaches the driver.

Federated preprocessing
-----------------------
Imputation and standardisation constants come from per-client SUFFICIENT
STATISTICS ONLY (count, sum, sum-of-squares per feature), combined on the
server by size-weighted averaging. No raw record leaves its client partition at
any point, including preprocessing. Mean imputation replaces median because the
mean is exactly federatable from sufficient statistics and the median is not.

Honest scope
------------
Clients are simulated partitions of one dataset inside one Spark cluster.
Parameters are exchanged rather than records, but NO differential privacy and
NO secure aggregation are implemented, so no formal privacy guarantee follows.
Tree ensembles are absent by necessity: decision trees cannot be
parameter-averaged.
"""
import argparse
import math
import time

import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (StructType, StructField, IntegerType,
                               StringType, LongType, DoubleType)

import config as C
from manifest import Run

LABEL = "hospital_expire_flag"
CONCEPTS = list(C.CHART_CONCEPTS.keys())

AGG_NUM = ["%s_%s" % (c, a) for c in CONCEPTS for a in C.CHART_AGGS]
MISS = ["%s_missing" % c for c in CONCEPTS]
NUMERIC = AGG_NUM + ["anchor_age"]          # mean-imputed then standardised
FEATURES = NUMERIC + MISS + ["gender_m"]    # fixed column order
D = len(FEATURES)

ADAPTIVE = ("fedadagrad", "fedadam", "fedyogi", "central_adam")
ONE_CLIENT = ("central", "central_adam")
PRED_COLS = ["seed", "condition", "model", "regime", "train_client",
             "eval_client", "split", "stay_id", "label", "p"]

PRED_SCHEMA = StructType([
    StructField("seed", IntegerType(), False),
    StructField("condition", StringType(), False),
    StructField("model", StringType(), False),
    StructField("regime", StringType(), False),
    StructField("train_client", StringType(), False),
    StructField("eval_client", StringType(), True),
    StructField("split", StringType(), False),
    StructField("stay_id", LongType(), False),
    StructField("label", DoubleType(), False),
    StructField("p", DoubleType(), False),
])


# ------------------------------------------------------------ parameter layout

def n_params(model, hidden):
    """Flat parameter vector length."""
    if model == "lr":
        return D + 1                                  # coefficients + intercept
    return D * hidden + hidden + hidden + 1           # W1, b1, W2, b2


def unpack(w, hidden):
    """Views into the flat vector: W1 (H,D), b1 (H,), W2 (H,), b2 scalar."""
    i = D * hidden
    W1 = w[:i].reshape(hidden, D)
    b1 = w[i:i + hidden]
    W2 = w[i + hidden:i + 2 * hidden]
    b2 = w[i + 2 * hidden]
    return W1, b1, W2, b2


def l2_mask(model, hidden):
    """True where L2 applies: weights yes, biases/intercept no."""
    P = n_params(model, hidden)
    m = np.zeros(P, dtype=bool)
    if model == "lr":
        m[:D] = True
        return m
    i = D * hidden
    m[:i] = True                                      # W1
    m[i + hidden:i + 2 * hidden] = True               # W2
    return m


def init_params(model, hidden, seed):
    """Shared initialisation. All clients MUST start from the same point.

    Zero initialisation is fine for logistic regression but fatal for an MLP:
    identical hidden units receive identical gradients and the symmetry never
    breaks. Glorot uniform, seeded from the run seed so the whole experiment
    stays reproducible."""
    P = n_params(model, hidden)
    if model == "lr":
        return np.zeros(P)
    rng = np.random.default_rng(seed)
    w = np.zeros(P)
    i = D * hidden
    lim1 = math.sqrt(6.0 / (D + hidden))
    lim2 = math.sqrt(6.0 / (hidden + 1))
    w[:i] = rng.uniform(-lim1, lim1, i)                       # W1
    w[i + hidden:i + 2 * hidden] = rng.uniform(-lim2, lim2, hidden)   # W2
    return w                                                   # biases stay 0


# ---------------------------------------------------------------- data loading

def load(spark, seed):
    """Feature matrix joined to splits. first_careunit is NEVER joined in: it is
    the partition key, so using it as a feature would leak the partition."""
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
    Only aggregates cross the client boundary."""
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
    """Cached RDD of (client_idx, label, feature array) for training."""
    cols = feature_exprs(stats)
    idx = F.create_map([F.lit(x) for kv in client_map.items() for x in kv])
    sel = df.select(idx[F.col(ccol)].alias("k"), F.col(LABEL).alias("y"),
                    F.array(*cols).alias("x"))
    rdd = (sel.rdd
           .map(lambda r: (int(r["k"]), float(r["y"]),
                           np.asarray(r["x"], dtype=np.float64)))
           .repartition(n_parts).cache())
    rdd.count()
    return rdd


def to_pred_rdd(df, stats, ccol, n_parts):
    """Cached RDD carrying identifiers, for scoring. Used for both model
    families so there is a single scoring path."""
    cols = feature_exprs(stats)
    sel = (df.filter(F.col("split").isin("val", "test"))
             .select(F.col("stay_id").cast("long").alias("stay_id"),
                     F.col("split"),
                     F.col(ccol).alias("eval_client"),
                     F.col(LABEL).alias("y"),
                     F.array(*cols).alias("x")))
    rdd = (sel.rdd
           .map(lambda r: (int(r["stay_id"]), r["split"], r["eval_client"],
                           float(r["y"]), np.asarray(r["x"], dtype=np.float64)))
           .repartition(n_parts).cache())
    rdd.count()
    return rdd


# ------------------------------------------------------------ distributed math

def _sigmoid(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _fwd_lr(w, x):
    return float(np.dot(w[:D], x)) + w[D]


def _fwd_mlp(w, x, hidden):
    W1, b1, W2, b2 = unpack(w, hidden)
    a1 = np.tanh(W1.dot(x) + b1)
    return float(W2.dot(a1) + b2), a1


def forward(w, x, model, hidden):
    if model == "lr":
        return _fwd_lr(w, x)
    z, _ = _fwd_mlp(w, x, hidden)
    return z


def grad_pass(rdd, sc, W, K, model, hidden):
    """One distributed pass. Returns (K, P+2):
       [:P] gradient sum, [P] n, [P+1] logloss sum.

    All K client gradients are computed in a SINGLE pass because each row looks
    up its own client's weights, so cost is O(rounds x local_epochs), not
    O(rounds x local_epochs x clients)."""
    bw = sc.broadcast(W)
    P = n_params(model, hidden)
    width = P + 2
    is_lr = (model == "lr")
    Hh = hidden

    def part(it):
        acc = np.zeros((K, width))
        Wl = bw.value
        for k, y, x in it:
            w = Wl[k]
            if is_lr:
                z = float(np.dot(w[:D], x)) + w[D]
                p = _sigmoid(z)
                g = p - y
                acc[k, :D] += g * x
                acc[k, D] += g
            else:
                W1, b1, W2, b2 = unpack(w, Hh)
                a1 = np.tanh(W1.dot(x) + b1)
                z = float(W2.dot(a1) + b2)
                p = _sigmoid(z)
                dz2 = p - y
                dz1 = (dz2 * W2) * (1.0 - a1 * a1)      # tanh derivative
                i = D * Hh
                acc[k, :i] += np.outer(dz1, x).ravel()  # dW1
                acc[k, i:i + Hh] += dz1                 # db1
                acc[k, i + Hh:i + 2 * Hh] += dz2 * a1   # dW2
                acc[k, i + 2 * Hh] += dz2               # db2
            acc[k, P] += 1.0
            pc = min(max(p, 1e-12), 1.0 - 1e-12)
            acc[k, P + 1] -= y * math.log(pc) + (1.0 - y) * math.log(1.0 - pc)
        yield acc

    out = rdd.mapPartitions(part).treeReduce(lambda a, b: a + b, depth=2)
    bw.destroy()
    return out


def collect_probs(rdd, sc, w, model, hidden):
    """DRIVER-SIDE ONLY -- a few thousand (p, y) pairs for a convergence check."""
    bw = sc.broadcast(w)

    def part(it):
        wl = bw.value
        for k, y, x in it:
            yield (_sigmoid(forward(wl, x, model, hidden)), y)

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

def server_update(strategy, w, delta, st, a, rnd):
    """Model-agnostic: operates on the flat parameter vector.

    FedAvg/FedProx take server_lr=1, where the update IS size-weighted
    averaging. Adaptive optimisers need a much smaller step: their update
    magnitude is ~server_lr_adaptive irrespective of pseudo-gradient scale.

    Reddi et al. (2021) omit Adam's bias correction, which makes early rounds
    crawl because m and v start at zero. We apply it, and say so, so the
    adaptive arms are not handicapped relative to FedAvg."""
    if strategy not in ADAPTIVE:
        return w + a.server_lr * delta
    slr = a.server_lr_adaptive
    if strategy == "fedadagrad":
        st["v"] += delta * delta
        return w + slr * delta / (np.sqrt(st["v"]) + a.tau)
    st["m"] = a.beta1 * st["m"] + (1.0 - a.beta1) * delta
    if strategy == "fedyogi":
        d2 = delta * delta
        st["v"] = st["v"] - (1.0 - a.beta2) * d2 * np.sign(st["v"] - d2)
    else:                                   # fedadam, central_adam
        st["v"] = a.beta2 * st["v"] + (1.0 - a.beta2) * delta * delta
    mh = st["m"] / (1.0 - a.beta1 ** rnd)
    vh = st["v"] / (1.0 - a.beta2 ** rnd)
    return w + slr * mh / (np.sqrt(np.maximum(vh, 0.0)) + a.tau)


def run_strategy(sc, strategy, train_rdd, val_rdd, sizes, K, a, tag, seed):
    """Multi-round federated training. Returns (weights, history)."""
    P = n_params(a.model, a.hidden)
    w = init_params(a.model, a.hidden, seed)
    st = {"m": np.zeros(P), "v": np.zeros(P)}
    mask = l2_mask(a.model, a.hidden)
    total = float(sum(sizes.values()))
    mu = a.mu if strategy == "fedprox" else 0.0
    hist = []
    t0 = time.time()

    for rnd in range(1, a.rounds + 1):
        W = {k: w.copy() for k in range(K)}
        train_loss = None

        for e in range(a.local_epochs):
            G = grad_pass(train_rdd, sc, W, K, a.model, a.hidden)
            if e == 0:
                n_all = G[:, P].sum()
                train_loss = float(G[:, P + 1].sum() / max(n_all, 1.0))
            for k in range(K):
                nk = G[k, P]
                if nk <= 0:
                    continue
                g = G[k, :P] / nk
                g[mask] += a.l2 * W[k][mask]
                if mu > 0.0:
                    g += mu * (W[k] - w)
                W[k] = W[k] - a.local_lr * g

        delta = np.zeros(P)
        for k in range(K):
            delta += (sizes[k] / total) * (W[k] - w)
        w = server_update(strategy, w, delta, st, a, rnd)

        rec = {"round": rnd, "train_loss": round(train_loss, 6),
               "delta_norm": round(float(np.linalg.norm(delta)), 6),
               "elapsed_s": round(time.time() - t0, 2)}
        if rnd % a.eval_every == 0 or rnd == a.rounds:
            pr = collect_probs(val_rdd, sc, w, a.model, a.hidden)
            rec["val_auroc"] = round(auroc(pr), 5)
            rec["val_logloss"] = round(
                float(np.mean([-(y * math.log(max(p, 1e-12)) +
                                 (1 - y) * math.log(max(1 - p, 1e-12)))
                               for p, y in pr])), 6)
            print("[%s] round %3d  train_loss=%.5f  val_auroc=%.4f  (%.1fs)"
                  % (tag, rnd, train_loss, rec["val_auroc"], rec["elapsed_s"]))
        hist.append(rec)

    return w, hist


def predict(spark, sc, pred_rdd, w, a, seed, condition, regime):
    """Score val+test through the RDD, in the S3 prediction schema. One scoring
    path for both model families."""
    bw = sc.broadcast(w)
    model, hidden = a.model, a.hidden
    tag = model + "_fl"

    def part(it):
        wl = bw.value
        for stay_id, split, ec, y, x in it:
            p = _sigmoid(forward(wl, x, model, hidden))
            yield (int(seed), condition, tag, regime, "federated",
                   ec, split, int(stay_id), float(y), float(p))

    out = spark.createDataFrame(pred_rdd.mapPartitions(part), schema=PRED_SCHEMA)
    return out.select(*PRED_COLS)


def main(a):
    seeds = [int(s) for s in a.seeds.split(",")]
    conditions = [c.strip() for c in a.conditions.split(",")]
    strategies = [s.strip() for s in a.strategies.split(",")]

    spark = SparkSession.builder.appName("s3b_%s" % a.model).getOrCreate()
    spark.conf.set("temporaryGcsBucket", C.BUCKET)
    spark.conf.set("spark.sql.shuffle.partitions", str(a.shuffle))
    sc = spark.sparkContext

    run = Run(spark, "s3b_fedavg", a.git_sha,
              {"model": a.model, "hidden": a.hidden if a.model == "mlp" else None,
               "n_params": n_params(a.model, a.hidden),
               "seeds": seeds, "conditions": conditions,
               "strategies": strategies, "rounds": a.rounds,
               "local_epochs": a.local_epochs, "local_lr": a.local_lr,
               "server_lr": a.server_lr,
               "server_lr_adaptive": a.server_lr_adaptive,
               "l2": a.l2, "mu": a.mu, "beta1": a.beta1, "beta2": a.beta2,
               "tau": a.tau, "n_features": D, "partitions": a.parts,
               "init": "zeros" if a.model == "lr" else "glorot-uniform, seeded",
               "activation": None if a.model == "lr" else "tanh",
               "preprocessing": "mean-impute + standardise from federated "
                                "sufficient statistics (count, sum, sumsq)",
               "privacy": "none -- no DP, no secure aggregation",
               "adam_bias_correction": True,
               "append": a.append, "no_write": a.no_write, "tag": a.tag})
    run.await_executors(a.expect_executors)

    out = C.PATH_MODELS + "/predictions_fl"
    h_uri = C.PATH_METRICS + "/fl_history" + ("" if a.model == "lr" else "_" + a.model)
    first = not a.append
    history, timing = [], []

    try:
        for seed in seeds:
            df = load(spark, seed).cache()
            df.count()

            for condition in conditions:
                ccol = "client_" + condition
                train = df.filter(F.col("split") == "train")
                stats, _ = federated_stats(train, ccol)

                names = sorted([r[0] for r in train.select(ccol).distinct().collect()])
                cmap = {nm: i for i, nm in enumerate(names)}
                sizes = {cmap[r[0]]: r[1] for r in
                         train.groupBy(ccol).count().collect()}

                tr_rdd = to_rdd(train, stats, cmap, ccol, a.parts)
                va_rdd = to_rdd(df.filter(F.col("split") == "val"),
                                stats, cmap, ccol, a.parts)
                pred_rdd = to_pred_rdd(df, stats, ccol, a.parts)

                # Single-client arms: identical code path, one client.
                tr_one = tr_rdd.map(lambda t: (0, t[1], t[2])).cache(); tr_one.count()
                va_one = va_rdd.map(lambda t: (0, t[1], t[2])).cache(); va_one.count()

                for strategy in strategies:
                    if strategy in ONE_CLIENT:
                        rdd_t, rdd_v, K = tr_one, va_one, 1
                        sz = {0: float(sum(sizes.values()))}
                    else:
                        rdd_t, rdd_v, K = tr_rdd, va_rdd, len(cmap)
                        sz = sizes

                    tag = "s%d/%s/%s/%s" % (seed, a.model, condition, strategy)
                    t0 = time.time()
                    w, hist = run_strategy(sc, strategy, rdd_t, rdd_v, sz, K,
                                           a, tag, seed)
                    dt = round(time.time() - t0, 2)

                    regime = "R3_" + strategy
                    for h in hist:
                        h.update({"seed": seed, "condition": condition,
                                  "strategy": strategy, "model": a.model})
                    history.extend(hist)
                    timing.append({"seed": seed, "model": a.model,
                                   "condition": condition, "strategy": strategy,
                                   "rounds": a.rounds,
                                   "passes": a.rounds * a.local_epochs,
                                   "n_clients": K, "wall_s": dt,
                                   "final_val_auroc": hist[-1].get("val_auroc")})

                    if a.no_write:
                        print("[skip] %s -> %s (%.1fs, --no-write)"
                              % (tag, regime, dt))
                    else:
                        preds = predict(spark, sc, pred_rdd, w, a, seed,
                                        condition, regime)
                        (preds.write.mode("overwrite" if first else "append")
                              .partitionBy("seed", "condition").parquet(out))
                        first = False
                        print("[write] %s -> %s (%.1fs)" % (tag, regime, dt))

                for r in (tr_rdd, va_rdd, pred_rdd, tr_one, va_one):
                    r.unpersist()
            df.unpersist()

        run.param("history", history)      # DRIVER-SIDE ONLY -- small table
        run.param("timing", timing)

        # --no-write leaves predictions_fl and fl_history untouched, so a
        # hyperparameter sweep cannot corrupt an existing result set. The
        # convergence history still lands in this run's manifest.
        if not a.no_write:
            run.output(out)
            run.count("prediction_rows", spark.read.parquet(out).count())
            spark.createDataFrame(history).coalesce(1) \
                 .write.mode("append" if a.append else "overwrite") \
                 .option("header", True).csv(h_uri)
            run.output(h_uri)
        else:
            print("[skip] no outputs written (--no-write)")

        spark.createDataFrame(timing).orderBy("seed", "condition",
                                              "strategy").show(80, False)
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
    p.add_argument("--model", default="lr", choices=["lr", "mlp"])
    p.add_argument("--hidden", type=int, default=16,
                   help="hidden units, --model mlp only")
    p.add_argument("--seeds", default="42")
    p.add_argument("--conditions", default="iid,careunit")
    p.add_argument("--strategies",
                   default="central,fedavg,fedprox,fedadam,fedadagrad")
    p.add_argument("--rounds", type=int, default=40)
    # E>1 is what produces client drift. At E=2 FedAvg collapses onto
    # centralised GD and no aggregation strategy can be distinguished.
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--local-lr", type=float, default=0.5)
    p.add_argument("--server-lr", type=float, default=1.0)
    p.add_argument("--server-lr-adaptive", type=float, default=1.0)
    p.add_argument("--l2", type=float, default=0.01)
    p.add_argument("--mu", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--shuffle", type=int, default=64)
    p.add_argument("--parts", type=int, default=8)
    p.add_argument("--append", action="store_true",
                   help="append to predictions_fl instead of overwriting")
    p.add_argument("--no-write", action="store_true",
                   help="tuning mode: log convergence to the manifest but write "
                        "no predictions and no history, so an existing result "
                        "set cannot be corrupted")
    p.add_argument("--tag", default="main")
    main(p.parse_args())