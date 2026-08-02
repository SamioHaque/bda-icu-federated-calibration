"""Central config. Shipped to Dataproc via --py-files; imported as `config`."""

PROJECT_ID = "innate-portal-501710-n1"
DATASET    = "icu_modeling"
BUCKET     = "innate-portal-501710-n1-spark"
REGION     = "us-east1"
ZONE       = "us-east1-b"
SEED       = 42

IMAGE_VERSION = "2.2-debian12"

# MIMIC-IV v3.1 copied into our own dataset (see README: Dataproc service accounts
# cannot inherit PhysioNet's user-level grant on physionet-data). Both aliases
# resolve to the same local dataset; kept distinct for readability at call sites.

MIMIC_ICU  = f"{PROJECT_ID}.{DATASET}"
MIMIC_HOSP = f"{PROJECT_ID}.{DATASET}"

GCS_ROOT      = f"gs://{BUCKET}"
PATH_COHORT   = f"{GCS_ROOT}/cohort"
PATH_SPLITS   = f"{GCS_ROOT}/splits"
PATH_FEATURES = f"{GCS_ROOT}/features"
PATH_MODELS   = f"{GCS_ROOT}/models"
PATH_METRICS  = f"{GCS_ROOT}/metrics"
PATH_MANIFEST = f"{GCS_ROOT}/manifests"


def bq(table: str) -> str:
    return f"{PROJECT_ID}.{DATASET}.{table}"

# Non-IID partition key. Groups MIMIC-IV's 17 first_careunit strings into three
# clinically coherent simulated hospitals. Counts below are all icustays rows
# (pre-cohort), verified against s_smoke output: 36,202 / 23,760 / 34,496 = 94,458.
CAREUNIT_GROUPS = {
    "Medical Intensive Care Unit (MICU)":                 "medical",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)":   "medical",
    "Medicine":                                           "medical",
    "Med/Surg":                                           "medical",
    "Intensive Care Unit (ICU)":                          "medical",
    "Surgical Intensive Care Unit (SICU)":                "surgical",
    "Trauma SICU (TSICU)":                                "surgical",
    "PACU":                                               "surgical",
    "Surgery/Vascular/Intermediate":                      "surgical",
    "Surgery/Trauma":                                     "surgical",
    "Cardiac Vascular Intensive Care Unit (CVICU)":       "cardiac_neuro",
    "Coronary Care Unit (CCU)":                           "cardiac_neuro",
    "Neuro Intermediate":                                 "cardiac_neuro",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)":    "cardiac_neuro",
    "Neuro Stepdown":                                     "cardiac_neuro",
    "Neurology":                                          "cardiac_neuro",
    "Medicine/Cardiology Intermediate":                   "cardiac_neuro",
}