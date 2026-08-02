"""Central config. Shipped to Dataproc via --py-files; imported as `config`."""

PROJECT_ID = "innate-portal-501710-n1"
DATASET    = "icu_modeling"
BUCKET     = "innate-portal-501710-n1-spark"
REGION     = "us-east1"
ZONE       = "us-east1-b"
SEED       = 42

IMAGE_VERSION = "2.2-debian12"

MIMIC_ICU  = "physionet-data.mimiciv_3_1_icu"
MIMIC_HOSP = "physionet-data.mimiciv_3_1_hosp"

GCS_ROOT      = f"gs://{BUCKET}"
PATH_COHORT   = f"{GCS_ROOT}/cohort"
PATH_SPLITS   = f"{GCS_ROOT}/splits"
PATH_FEATURES = f"{GCS_ROOT}/features"
PATH_MODELS   = f"{GCS_ROOT}/models"
PATH_METRICS  = f"{GCS_ROOT}/metrics"
PATH_MANIFEST = f"{GCS_ROOT}/manifests"


def bq(table: str) -> str:
    return f"{PROJECT_ID}.{DATASET}.{table}"