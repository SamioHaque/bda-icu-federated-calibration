# Automation & Reporting Scripts (`scripts/`)

This directory contains Python scripts for post-processing results, generating figures and paper tables, alongside PowerShell automation scripts for GCP Dataproc cluster management.

---

## Python Reporting & Visualization Tools

* **`figures.py`**: Generates publication-ready figures including Calibration Curves, ROC curves, PR curves, and federated optimization convergence plots.
* **`report.py`**: Parses evaluation JSON metrics and manifests to compile paper summary tables (LaTeX & Markdown).
* **`fig_pipeline.py`**: Generates a schematic architecture diagram of the distributed federated workflow.
* **`scaling_table.py`**: Formats scalability runtime and efficiency measurements across varying Spark cluster node scales.
* **`make_synthetic_example.py`**: Generates a 200-row synthetic dataset (`results/synthetic/SYNTHETIC_feature_matrix_200rows.csv`) for dry-running and testing the pipeline without MIMIC-IV credentialed data.

---

## PowerShell Cloud Infrastructure Automation (`.ps1`)

* **`cluster_create.ps1`**: Provisions a Google Cloud Dataproc Spark cluster using settings configured in `config.py`.
* **`cluster_delete.ps1`**: Deletes the active GCP Dataproc cluster to prevent unnecessary cloud resource consumption.
* **`submit.ps1`**: Submits PySpark jobs from `src/` to the running GCP Dataproc cluster.
* **`pull_results.ps1`**: Downloads processed metric JSON files and log manifests from Google Cloud Storage to local disk.
* **`sweep_scalability.ps1`**: Executes batch parameter sweeps across scaling configurations and client node combinations.
