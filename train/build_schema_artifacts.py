"""Emit the artifacts the schema-driven frontend needs.

  artifacts/feature_schema.json  per-feature type, range, median, missingness
  artifacts/examples.json        synthetic demo patients

Only aggregate statistics leave the training data. No real MIMIC-IV record is
written to either file: redistributing row-level data would breach the PhysioNet
data-use agreement, so the examples are synthetic archetypes assembled from
plausible clinical values.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parent
PUB = API_ROOT.parent
RESULTS = PUB / "medicare-30day-readmission-mimic-iv" / "results"
DATA = PUB / "Dataset" / "mimic-parquet"
ART = API_ROOT / "artifacts"

import sys
sys.path.insert(0, str(API_ROOT))
from app.config import client_facing_fields  # noqa: E402

art = joblib.load(ART / "serving_artifacts.joblib")
FEATURES = art["feature_order"]
CLIENT_FACING = client_facing_fields(FEATURES)
CAT_MAPS = art["cat_maps"]

V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
V10 = pd.read_parquet(PUB / "training_table_v10.parquet")
v7u = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7u].reset_index(drop=True)], axis=1)
race = pd.read_parquet(DATA / "admissions.parquet", columns=["hadm_id", "race"])
DF = DF.merge(race, on="hadm_id", how="left")

split = np.load(RESULTS / "v7_split_indices.npz")
train = DF.iloc[split["train_idx"]]

schema = []
for name in CLIENT_FACING:
    col = train[name]
    pct_nan = float(col.isna().mean())
    if name in CAT_MAPS:
        levels = sorted(CAT_MAPS[name].keys())
    elif name == "race":
        levels = sorted(art["te_maps"]["race_te"].keys())
    elif name == "last_drg_dispo":
        levels = sorted(CAT_MAPS.get("last_drg_dispo", {}).keys())
    else:
        levels = None

    if levels is not None:
        schema.append({"name": name, "type": "categorical",
                       "levels": levels, "pct_nan": round(pct_nan, 6)})
    else:
        s = pd.to_numeric(col, errors="coerce")
        schema.append({
            "name": name, "type": "numeric",
            "min": float(np.nanmin(s)) if s.notna().any() else 0.0,
            "median": float(np.nanmedian(s)) if s.notna().any() else 0.0,
            "max": float(np.nanmax(s)) if s.notna().any() else 0.0,
            "pct_nan": round(pct_nan, 6),
        })

(ART / "feature_schema.json").write_text(json.dumps(schema, indent=2))
print(f"feature_schema.json: {len(schema)} client-facing features "
      f"({sum(1 for f in schema if f['type']=='categorical')} categorical)")

# ---------------------------------------------------------------- examples
# Synthetic archetypes. Values are clinically plausible and were chosen by hand;
# none is copied from a MIMIC-IV record.
EXAMPLES = [
    {   # low risk: elective surgical admission, no history
        "age_at_admit": 68, "los_days": 2.1, "admission_type": "ELECTIVE",
        "admission_location": "PHYSICIAN REFERRAL", "discharge_location": "HOME",
        "primary_dx_chapter": "M", "drg_code": "470", "n_diagnoses": 5,
        "prior_admissions_6m": 0, "prior_admissions_all": 0,
        "prior_readmission_count": 0, "time_since_last_discharge": 365,
        "race": "WHITE", "creatinine_last": 0.9, "bun_last": 14,
        "sodium_last": 140, "hemoglobin_last": 12.8, "albumin_last": 4.0,
        "n_discharge_drugs": 5, "n_meds_total": 12, "n_distinct_providers": 4,
    },
    {   # moderate risk: heart failure, some history, home health
        "age_at_admit": 76, "los_days": 5.4, "admission_type": "EW EMER.",
        "admission_location": "EMERGENCY ROOM", "discharge_location": "HOME HEALTH CARE",
        "primary_dx_chapter": "I", "drg_code": "291", "n_diagnoses": 12,
        "prior_admissions_6m": 1, "prior_admissions_all": 3,
        "prior_readmission_count": 1, "time_since_last_discharge": 45,
        "race": "WHITE", "creatinine_last": 1.4, "bun_last": 28,
        "sodium_last": 136, "hemoglobin_last": 9.8, "albumin_last": 3.1,
        "n_discharge_drugs": 14, "n_meds_total": 38, "n_distinct_providers": 8,
    },
    {   # high risk: frequent utiliser, recent discharge, skilled nursing
        "age_at_admit": 81, "los_days": 9.2, "admission_type": "EW EMER.",
        "admission_location": "EMERGENCY ROOM",
        "discharge_location": "SKILLED NURSING FACILITY",
        "primary_dx_chapter": "I", "drg_code": "291", "n_diagnoses": 18,
        "prior_admissions_6m": 4, "prior_admissions_all": 11,
        "prior_readmission_count": 3, "time_since_last_discharge": 12,
        "race": "BLACK/AFRICAN AMERICAN", "creatinine_last": 1.9, "bun_last": 42,
        "sodium_last": 132, "hemoglobin_last": 8.6, "albumin_last": 2.6,
        "n_discharge_drugs": 19, "n_meds_total": 61, "n_distinct_providers": 14,
        "los_trend_180d": 1.8, "prior_mean_los_6m": 7.5,
    },
    {   # oncology, metastatic disease
        "age_at_admit": 72, "los_days": 6.8, "admission_type": "URGENT",
        "admission_location": "TRANSFER FROM HOSPITAL",
        "discharge_location": "HOME HEALTH CARE",
        "primary_dx_chapter": "C", "drg_code": "846", "n_diagnoses": 15,
        "prior_admissions_6m": 2, "prior_admissions_all": 6,
        "prior_readmission_count": 2, "time_since_last_discharge": 28,
        "race": "ASIAN", "creatinine_last": 1.1, "bun_last": 22,
        "sodium_last": 134, "hemoglobin_last": 9.1, "albumin_last": 2.9,
        "bilirubin_max": 2.4, "elix_mets": 1, "elix_solid_tumor": 1,
        "n_discharge_drugs": 16, "n_meds_total": 44, "n_distinct_providers": 11,
    },
    {   # respiratory, observation stay
        "age_at_admit": 70, "los_days": 3.3, "admission_type": "OBSERVATION ADMIT",
        "admission_location": "EMERGENCY ROOM", "discharge_location": "HOME",
        "primary_dx_chapter": "J", "drg_code": "190", "n_diagnoses": 9,
        "prior_admissions_6m": 1, "prior_admissions_all": 2,
        "prior_readmission_count": 0, "time_since_last_discharge": 120,
        "race": "HISPANIC OR LATINO", "creatinine_last": 1.0, "bun_last": 18,
        "sodium_last": 138, "hemoglobin_last": 11.9, "albumin_last": 3.5,
        "n_discharge_drugs": 9, "n_meds_total": 21, "n_distinct_providers": 6,
    },
]
(ART / "examples.json").write_text(json.dumps(EXAMPLES, indent=2))
print(f"examples.json: {len(EXAMPLES)} synthetic archetypes")
