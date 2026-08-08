"""Model loading and inference, including per-patient SHAP attributions.

Artifacts load once and are reused for every request. XGBoost's native TreeSHAP
(`pred_contribs`) gives exact attributions with no extra dependency and sub-millisecond
overhead, so every explanation is computed on demand rather than approximated.
"""
from __future__ import annotations

import json
import math
import threading
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import xgboost as xgb

from .config import AFT_META_PATH, AFT_MODEL_PATH, METADATA_PATH, MODEL_PATH, SERVING_PATH
from .features import build_vector

_lock = threading.Lock()


class Predictor:
    def __init__(self) -> None:
        self.booster = xgb.Booster()
        self.booster.load_model(str(MODEL_PATH))
        self.artifacts: dict[str, Any] = joblib.load(SERVING_PATH)
        self.metadata: dict[str, Any] = json.loads(METADATA_PATH.read_text())
        self.feature_order: list[str] = self.artifacts["feature_order"]
        self.threshold: float = float(self.artifacts["threshold"])
        self.medians: dict[str, float] = self.artifacts["medians"]
        self.tiers: dict[str, float] = self.artifacts["tiers"]
        seed = self.metadata.get("seed", 42)
        self.model_name: str = f"xgboost-rfe{len(self.feature_order)}-seed{seed}"

        # Optional AFT survival booster for day-by-day trajectories. Loaded once,
        # alongside the binary model; its absence must never break the classifier.
        self.aft_booster: xgb.Booster | None = None
        self.aft_meta: dict[str, Any] = {}
        if AFT_MODEL_PATH.exists():
            self.aft_booster = xgb.Booster()
            self.aft_booster.load_model(str(AFT_MODEL_PATH))
            if AFT_META_PATH.exists():
                self.aft_meta = json.loads(AFT_META_PATH.read_text())
        self.aft_sigma: float = float(self.aft_meta.get("sigma", 1.0))
        self.aft_model_name: str = f"xgboost-aft{len(self.feature_order)}-seed{seed}"

    @property
    def trajectory_available(self) -> bool:
        return self.aft_booster is not None

    def _warnings(self, imputed: list[str]) -> list[str]:
        """Human-readable notes about every value the server had to assume.

        A caller should be able to see exactly which numbers were theirs and which
        came from the training distribution. For categorical features the imputed
        value is an encoded index, which would read as a nonsense number (e.g.
        "median (4333)"), so those get a wording without the value.
        """
        cat = set(self.artifacts.get("cat_maps", {}))
        out = []
        for f in sorted(imputed):
            med = self.medians.get(f)
            if f in cat or med is None:
                out.append(f"'{f}' was not supplied; a typical category was assumed.")
            else:
                out.append(f"'{f}' was not supplied; using the training median ({med:g}).")
        return out

    def predict(self, payload: dict | None, with_shap: bool = False) -> dict:
        x, imputed = build_vector(payload or {}, self.artifacts)
        dm = xgb.DMatrix(x, feature_names=self.feature_order)

        with _lock:
            prob = float(self.booster.predict(dm)[0])
            contribs = (
                self.booster.predict(dm, pred_contribs=True)[0] if with_shap else None
            )

        result: dict[str, Any] = {
            "probability": round(prob, 6),
            "fallback_warnings": self._warnings(imputed),
            "imputed_fields": sorted(imputed),
        }

        if contribs is not None:
            # the final entry of pred_contribs is the bias term, on the log-odds scale
            result["shap_values"] = [round(float(v), 6) for v in contribs[:-1]]
            result["base_value"] = round(float(contribs[-1]), 6)
            result["feature_values"] = [round(float(v), 6) for v in x[0]]

        return result

    def predict_trajectory(self, payload: dict | None, horizon: int = 30) -> dict:
        """Day-by-day cumulative readmission probability from the AFT model.

        XGBoost AFT with a normal loss predicts on the TIME scale, so the raw
        prediction is exp(mu). Under the lognormal AFT assumption the cumulative
        probability of readmission by day t is Phi((ln t - mu) / sigma), which is
        non-decreasing in t by construction.
        """
        if self.aft_booster is None:
            raise RuntimeError("Trajectory model artifact is not available.")

        x, imputed = build_vector(payload or {}, self.artifacts)
        dm = xgb.DMatrix(x, feature_names=self.feature_order)
        with _lock:
            pred_time = float(self.aft_booster.predict(dm)[0])

        mu = math.log(max(pred_time, 0.05))
        denom = self.aft_sigma * math.sqrt(2.0)

        def cdf(t: float) -> float:
            return 0.5 * (1.0 + math.erf((math.log(t) - mu) / denom))

        cumulative = [round(min(max(cdf(t), 0.0), 1.0), 6) for t in range(1, horizon + 1)]
        for i in range(1, len(cumulative)):           # guard against rounding noise
            cumulative[i] = max(cumulative[i], cumulative[i - 1])
        increments = [cumulative[0]] + [
            round(cumulative[i] - cumulative[i - 1], 6) for i in range(1, len(cumulative))
        ]

        return {
            "days": list(range(1, horizon + 1)),
            "cumulative_probability": cumulative,
            "daily_increment": increments,
            "median_predicted_day": round(math.exp(mu), 1),
            "horizon_probability": cumulative[-1],
            "fallback_warnings": self._warnings(imputed),
            "imputed_fields": sorted(imputed),
        }


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()
