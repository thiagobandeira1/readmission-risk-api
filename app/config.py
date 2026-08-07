"""Feature contract: which of the 67 model features come from the client and which
are derived server-side. Keeping this in one place means the API, the docs and the
Lovable client can never drift apart."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.json"
SERVING_PATH = ARTIFACT_DIR / "serving_artifacts.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

API_TITLE = "30-Day Readmission Risk API"
API_VERSION = "1.0.0"

# --- features the server computes; never accepted from the client -------------
# 11 verified algebraic derivations (each reproduces the training column exactly)
DERIVED_FEATURES = [
    "albumin_x_los",                # albumin_last * los_days
    "los_x_n_diagnoses",            # los_days * n_diagnoses
    "n_meds_x_age",                 # age_at_admit * n_discharge_drugs
    "creatinine_x_bun",             # creatinine_last * bun_last
    "bun_creatinine_ratio",         # bun_last / creatinine_last
    "los_trend_x_prior_6m",         # los_trend_180d * prior_admissions_6m
    "los_trend_x_prior_admits",     # los_trend_180d * prior_admissions_all
    "severity_x_lab_abnormal",      # severity_composite * lab_abnormal_rate
    "log_time_since_discharge",     # log1p(time_since_last_discharge)
    "los_per_prior_admit",          # los_days / (prior_admissions_all + 1)
    "provider_fragmentation_x_los",  # los_days * n_distinct_providers
]
# 5 target encodings, resolved through maps fitted on the training split only
TE_FEATURES = [
    "primary_dx_chapter_te", "discharge_location_te", "drg_code_te",
    "last_drg_dispo_te", "race_te",
]
SERVER_COMPUTED = set(DERIVED_FEATURES) | set(TE_FEATURES)

# --- clinically essential; the request is rejected without them ---------------
REQUIRED_FIELDS = [
    "age_at_admit", "los_days", "admission_type", "admission_location",
    "discharge_location", "primary_dx_chapter", "drg_code", "n_diagnoses",
    "prior_admissions_6m", "prior_admissions_all", "prior_readmission_count",
    "time_since_last_discharge",
]

# --- inputs used only to build derived features, never fed to the model raw ---
DERIVATION_ONLY_INPUTS = ["n_diagnoses", "creatinine_last", "race"]

# Any optional field left null is filled with the training-set median and named
# in the response's `imputed_fields`, so a caller always knows what was assumed.
RISK_TIER_LABELS = ["Low", "Moderate", "High", "Very High"]
