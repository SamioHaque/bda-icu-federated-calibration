# Spark-Based Distributed EHR Modeling for Calibrated ICU Risk

MIMIC-IV v3.1 in-hospital mortality from the first 24 ICU hours.
GCP Dataproc image 2.2-debian12, Spark 3.5.3, seed 42/43/44.

## Scale
chartevents rows scanned: 432,997,491
measurements aggregated:   10,736,029
cohort: 65,366 first ICU stays, 7,086 deaths (10.84% prevalence)

## Pipeline order
s0_cohort -> s1_splits -> s2a_itemids -> s2_features
  -> s3_train (MLlib: R1 pooled, R2 client-local, R2e prediction ensemble)
  -> s3b_fedavg (federated: central, FedAvg, FedProx, FedAdam, FedAdagrad)
  -> s4_evaluate (isotonic calibration, two scopes; all metrics)
  -> scripts/report.py (paper tables, local)

Every job writes a manifest to gs://.../manifests/ with git SHA, cluster
shape, row counts and wall clock. All reported numbers derive from those.

## Frozen settings
LR:  maxIter 100, regParam 0.01, no class weighting
GBT: maxIter 30, maxDepth 5, stepSize 0.1
FL:  40 rounds, 5 local epochs, local_lr 0.5, server_lr 1.0,
     adaptive server_lr 1.0, tau 1.0, mu 0.1 (FedProx)
Splits: 60/20/20 by hash of subject_id; test at natural prevalence.

## Honest scope
- Clients are simulated partitions of one dataset, not real hospitals.
- No differential privacy, no secure aggregation: raw records never leave
  a partition, but no formal privacy guarantee is claimed.
- GBT cannot be parameter-averaged; federated arms are LR only.
- Hyperparameters are fixed, not tuned (except the FedAdam server step,
  swept over {0.01, 0.03, 0.1, 0.3} and tau over {1e-3, 0.1, 1.0}).
