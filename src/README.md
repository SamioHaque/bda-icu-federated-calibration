# Source Pipeline (`src/`)

This directory contains the core PySpark distributed pipeline for data cohort extraction, feature engineering, model training, federated optimization, and evaluation.

## Pipeline Overview & Stage Order

The data processing and execution follow a strict sequential order:

```
s0_cohort -> s1_splits -> s2a_itemids -> s2_features -> s3_train / s3b_fedavg -> s4_evaluate -> s5_scalability
```

---

## File Breakdown

### 1. Cohort & Data Preparation
* **`s0_cohort.py`**: Queries MIMIC-IV v3.1 from BigQuery/GCS to extract the target cohort (first ICU stay, age ≥ 18, ICU duration ≥ 24 hours, 24-hour in-hospital mortality target label).
* **`s1_splits.py`**: Performs hash-based train/validation/test splits (60/20/20 ratio by `subject_id`) and partitions data into simulated hospital client sites.
* **`s2a_itemids.py`**: Maintains dictionary mappings and MIMIC-IV item IDs for vital signs, laboratory measurements, and clinical chart events.
* **`s2_features.py`**: Aggregates time-series clinical events during the first 24 hours of ICU stay into summary statistics (min, max, mean, count) to create tabular Spark FeatureVectors.

### 2. Modeling & Federated Optimization
* **`s3_train.py`**: Trains baseline centralized models (Logistic Regression and Gradient Boosted Trees) on pooled datasets as well as client-local models using Spark MLlib.
* **`s3b_fedavg.py`**: Implements federated optimization algorithms across simulated client sites:
  * **FedAvg** (Standard Federated Averaging)
  * **FedProx** (Handles client heterogeneity via proximal terms)
  * **FedAdam** / **FedAdagrad** (Adaptive federated server optimizers)

### 3. Evaluation & Benchmarking
* **`s4_evaluate.py`**: Computes discrimination metrics (AUROC, AUPRC) and probability calibration metrics (Brier Score, Expected Calibration Error (ECE)), applying Isotonic Regression calibration across pooled and site-specific scopes.
* **`s5_scalability.py`**: Evaluates system throughput, computational speedup, and wall-clock execution time across different cluster node counts.
* **`s_smoke.py`**: Provides a fast end-to-end sanity check / dry-run on synthetic data.

### 4. Utilities & Infrastructure
* **`manifest.py`**: Handles experiment auditing, tracking execution parameters, dataset row counts, cluster configuration, Git commit SHA, and runtime metrics saved to GCS.
