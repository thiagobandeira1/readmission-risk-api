"""Model loading and inference, including per-patient SHAP attributions.

Artifacts load once at import and are reused for every request; XGBoost's native
TreeSHAP (`pred_contribs`) gives exact attributions with no extra dependency and
sub-millisecond overhead, so every prediction can ship its own explanation.
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import xgboost as xgb

from .config import METADATA_PATH, MODEL_PATH, RISK_TIER_LABELS, SERVING_PATH
from .features import build_vector

_lock = threading.Lock()

# Human-readable labels so the UI never has to show a raw column name.
FEATURE_LABELS: dict[str, str] = {
    "los_trend_180d": "Length-of-stay trend (180 days)",
    "discharge_location_te": "Discharge destination (risk-encoded)",
    "discharge_location": "Discharge destination",
    "prior_admissions_6m": "Prior admissions (6 months)",
    "prior_admissions_all": "Prior admissions (all time)",
    "prior_readmission_count": "Prior readmissions",
    "time_since_last_discharge": "Days since last discharge",
    "log_time_since_discharge": "Days since last discharge (log)",
    "drg_code_te": "DRG code (risk-encoded)",
    "drg_code": "DRG code",
    "primary_dx_chapter_te": "Primary diagnosis chapter (risk-encoded)",
    "primary_dx_chapter": "Primary diagnosis chapter",
    "age_at_admit": "Age at admission",
    "los_days": "Length of stay (days)",
    "albumin_last": "Albumin (last)",
    "bun_last": "BUN (last)",
    "sodium_last": "Sodium (last)",
    "hemoglobin_last": "Hemoglobin (last)",
    "wbc_last": "White blood cell count (last)",
    "glucose_last": "Glucose (last)",
    "bicarbonate_last": "Bicarbonate (last)",
    "bilirubin_max": "Bilirubin (max)",
    "bmi_last": "BMI",
    "severity_composite": "Severity composite",
    "clinical_complexity": "Clinical complexity",
    "lab_abnormal_rate": "Abnormal-lab rate",
    "n_meds_total": "Total medications",
    "n_discharge_drugs": "Discharge medications",
    "distinct_drugs": "Distinct drugs",
    "n_procedures": "Procedures",
    "n_diagnoses": "Diagnoses",
    "elix_mets": "Metastatic cancer",
    "elix_solid_tumor": "Solid tumour",
    "elix_psychoses": "Psychoses",
    "race_te": "Race (risk-encoded)",
}


class Predictor:
    def __init__(self) -> None:
        self.booster = xgb.Booster()
        self.booster.load_model(str(MODEL_PATH))
        self.artifacts: dict[str, Any] = joblib.load(SERVING_PATH)
        self.metadata: dict[str, Any] = json.loads(METADATA_PATH.read_text())
        self.feature_order: list[str] = self.artifacts["feature_order"]
        self.threshold: float = float(self.artifacts["threshold"])
        self.tiers: dict[str, float] = self.artifacts["tiers"]

    # -- helpers ----------------------------------------------------------
    def risk_tier(self, p: float) -> str:
        t = self.tiers
        if p <= t["low_max"]:
            return RISK_TIER_LABELS[0]
        if p <= t["moderate_max"]:
            return RISK_TIER_LABELS[1]
        if p <= t["high_max"]:
            return RISK_TIER_LABELS[2]
        return RISK_TIER_LABELS[3]

    @staticmethod
    def label(feature: str) -> str:
        if feature in FEATURE_LABELS:
            return FEATURE_LABELS[feature]
        return feature.replace("_", " ").replace(" te", " (risk-encoded)").capitalize()

    # -- inference --------------------------------------------------------
    def predict(self, payload: dict, top_k: int = 8) -> dict:
        x, imputed = build_vector(payload, self.artifacts)
        dm = xgb.DMatrix(x, feature_names=self.feature_order)

        with _lock:
            prob = float(self.booster.predict(dm)[0])
            contribs = self.booster.predict(dm, pred_contribs=True)[0]

        # last entry of pred_contribs is the bias/base value (log-odds scale)
        shap_vals = contribs[:-1]
        base = float(contribs[-1])
        order = np.argsort(np.abs(shap_vals))[::-1][:top_k]

        imputed_set = set(imputed)
        drivers = [
            {
                "feature": self.feature_order[i],
                "label": self.label(self.feature_order[i]),
                "value": float(x[0, i]),
                "contribution": float(shap_vals[i]),
                "direction": "increases" if shap_vals[i] > 0 else "decreases",
                # True when this value was assumed rather than supplied: the UI must
                # not present an imputed field as if it were an observed finding.
                "imputed": self.feature_order[i] in imputed_set,
            }
            for i in order
        ]

        return {
            "readmission_probability": round(prob, 6),
            "risk_tier": self.risk_tier(prob),
            "flagged": bool(prob >= self.threshold),
            "threshold": round(self.threshold, 6),
            "base_rate_log_odds": round(base, 6),
            "top_drivers": drivers,
            "imputed_fields": sorted(imputed),
            "n_features_used": len(self.feature_order),
            "model_version": self.metadata.get("feature_set", "RFE-67"),
        }


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()
