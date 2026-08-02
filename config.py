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

# --- chartevents feature concepts ------------------------------------------
# Verified against d_items by s2a_itemids (manifest s2a_itemids-20260802T094010).
# Multiple recording routes per concept are POOLED, not treated as separate
# features: no patient carries all routes, so separate columns would be mostly
# null. Invasive and cuff pressures differ systematically -- noted as a
# limitation; per-route counts can be profiled separately if needed.
CHART_CONCEPTS = {
    "hr":   [220045],
    "sbp":  [220050, 225309, 220179, 224167, 227243],   # arterial, ART line, NIBP, manual L/R
    "dbp":  [220051, 225310, 220180, 224643, 227242],
    "map":  [220052, 225312, 220181],
    "rr":   [220210],                                    # excludes ventilator Set/spont/Total
    "spo2": [220277],                                    # excludes 220227 SaO2 (blood gas)
    "temp": [223762, 223761],                            # C native; F converted in S2
}
TEMP_F_ITEMID = 223761

# Physiologically implausible values are artifacts (transducer flush, cuff
# error). Dropped before aggregation, since min/max are the aggregates most
# sensitive to a single bad reading.
CHART_RANGES = {
    "hr": (20.0, 300.0), "sbp": (30.0, 300.0), "dbp": (10.0, 200.0),
    "map": (20.0, 250.0), "rr": (0.0, 80.0), "spo2": (50.0, 100.0),
    "temp": (25.0, 45.0),
}
CHART_AGGS = ["min", "max", "mean", "std", "first", "last", "count"]