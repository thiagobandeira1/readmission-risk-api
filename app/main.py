"""FastAPI application exposing the 30-day readmission model.

The route and payload shapes match the RiskPath Console frontend contract:

    GET  /health                          liveness + model identifier
    GET  /metadata                        feature schema, model info, threshold
    GET  /examples?n=5                    synthetic demo patients
    POST /predictions?threshold=0.2       one patient  -> probability + class
    POST /explanations                    one patient  -> SHAP attributions
    POST /predictions/batch?threshold=0.2 many patients

Requests carry a flat `{feature: value}` object. The server derives the
interactions, log transforms and target encodings the model needs, so the client
never has to know the model's internal feature engineering. Any field left out is
filled from the training median and reported back in `fallback_warnings`.

Errors use the envelope `{"error": {"code", "message", "details"}}`.
"""
from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import API_TITLE, API_VERSION, ARTIFACT_DIR, REQUIRED_FIELDS
from .predictor import get_predictor

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Predicts 30-day all-cause hospital readmission risk for Medicare patients "
        "from structured EHR data. XGBoost over 66 features selected by recursive "
        "feature elimination (race excluded) on the MIMIC-IV v3.1 Medicare cohort (244,576 admissions).\n\n"
        "**Research prototype — not a medical device.** Intended to help prioritise "
        "transitional-care resources, never to withhold or reduce care. Not externally "
        "validated or prospectively evaluated."
    ),
)

_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_origin_regex=r"https://.*\.(lovable\.app|lovableproject\.com|pages\.dev|workers\.dev)|http://localhost:\d+|http://127\.0\.0\.1:\d+",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_FEATURE_SCHEMA = json.loads((ARTIFACT_DIR / "feature_schema.json").read_text())
_EXAMPLES = json.loads((ARTIFACT_DIR / "examples.json").read_text())


# --------------------------------------------------------------- error envelope
def _envelope(code: str, message: str, details: dict | None = None) -> dict:
    body: dict[str, Any] = {"code": code, "message": message}
    if details:
        body["details"] = details
    return {"error": body}


@app.exception_handler(RequestValidationError)
async def on_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=_envelope("VALIDATION_ERROR", "The request payload is invalid.",
                          {"errors": exc.errors()[:20]}),
    )


@app.exception_handler(Exception)
async def on_unhandled(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=_envelope("INTERNAL_ERROR", str(exc) or "Unexpected server error."),
    )


# ------------------------------------------------------------------- endpoints
@app.get("/health", tags=["ops"])
def health() -> dict:
    p = get_predictor()
    return {"status": "ok", "model": p.model_name}


@app.get("/metadata", tags=["ops"])
def metadata() -> dict:
    """Feature schema that drives the frontend form, plus the model card.

    `features` lists only what a caller supplies; the 16 features the server derives
    are deliberately absent so the UI never asks for a target encoding.
    """
    p = get_predictor()
    m = p.metadata["metrics"]
    return {
        "features": _FEATURE_SCHEMA,
        "model_info": {
            "name": p.model_name,
            "seed": p.metadata.get("seed", 42),
            "n_features": len(p.feature_order),
            "published_test_auroc": 0.7957,          # V8 value reported in the manuscript
            "deployed_test_auroc": m["test_auroc"],  # this artifact, re-measured
        },
        "default_threshold": p.threshold,
        # Empirical cut-points from validation-set quantiles (50th/80th/95th).
        # Fixed bands like 0.30/0.60/0.85 do not suit this model: with a 21%
        # base rate its 95th percentile is only ~0.58, so a fixed scale would
        # label almost every patient "low" and never use the top band.
        "risk_bands": {
            "low_max": p.tiers["low_max"],
            "moderate_max": p.tiers["moderate_max"],
            "high_max": p.tiers["high_max"],
            "basis": "validation-set quantiles (p50 / p80 / p95)",
        },
        "trajectory_available": p.trajectory_available,
        "required_fields": REQUIRED_FIELDS,
        "disclaimer": (
            "Research prototype trained on MIMIC-IV v3.1. Not externally validated, "
            "not prospectively evaluated, and not a medical device."
        ),
    }


@app.get("/examples", tags=["ops"])
def examples(n: int = Query(5, ge=1, le=50)) -> dict:
    """Synthetic demo patients.

    These are hand-built archetypes, not MIMIC-IV records: redistributing
    row-level data would breach the PhysioNet data-use agreement.
    """
    subset = _EXAMPLES[:n]
    return {"examples": subset, "n": len(subset)}


@app.post("/predictions", tags=["inference"])
def predictions(features: dict, threshold: float | None = Query(None, ge=0.0, le=1.0)) -> dict:
    p = get_predictor()
    thr = p.threshold if threshold is None else float(threshold)
    r = p.predict(features)
    return {
        "probability": r["probability"],
        "prediction": int(r["probability"] >= thr),
        "threshold": thr,
        "model_name": p.model_name,
        "fallback_warnings": r["fallback_warnings"],
    }


@app.post("/explanations", tags=["inference"])
def explanations(features: dict) -> dict:
    p = get_predictor()
    r = p.predict(features, with_shap=True)
    return {
        "shap_values": r["shap_values"],
        "base_value": r["base_value"],
        "feature_names": p.feature_order,
        "feature_values_transformed": r["feature_values"],
        "probability": r["probability"],
        "model_name": p.model_name,
        "fallback_warnings": r["fallback_warnings"],
    }


@app.post("/predictions/trajectory", tags=["inference"])
def predictions_trajectory(features: dict) -> dict:
    """Cumulative probability of readmission on each day of the 30-day window.

    Answers *when*, not just *whether*: the binary endpoints return one number for
    the whole window, this one returns the risk trajectory across it.
    """
    p = get_predictor()
    if not p.trajectory_available:
        return JSONResponse(
            status_code=503,
            content=_envelope("INTERNAL_ERROR", "Trajectory model is not loaded on this server."),
        )
    r = p.predict_trajectory(features)
    r["model_name"] = p.aft_model_name
    r["disclaimer"] = ("Parametric AFT estimate; deaths treated as censoring; "
                       "research prototype.")
    return r


@app.post("/predictions/batch", tags=["inference"])
def predictions_batch(payload: dict, threshold: float | None = Query(None, ge=0.0, le=1.0)) -> dict:
    p = get_predictor()
    thr = p.threshold if threshold is None else float(threshold)
    patients = payload.get("patients") or []
    if not isinstance(patients, list):
        return JSONResponse(
            status_code=422,
            content=_envelope("VALIDATION_ERROR", "`patients` must be a list of feature objects."),
        )
    if len(patients) > 1000:
        return JSONResponse(
            status_code=422,
            content=_envelope("VALIDATION_ERROR", "At most 1000 patients per batch request."),
        )

    results, warnings = [], []
    for row in patients:
        # a batch row may arrive either flat or wrapped as {"features": {...}}
        feats = row.get("features") if isinstance(row, dict) and "features" in row else row
        r = p.predict(feats or {})
        results.append({"probability": r["probability"],
                        "prediction": int(r["probability"] >= thr)})
        warnings.extend(r["fallback_warnings"])

    # collapse duplicate warnings across the batch
    seen, unique = set(), []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    return {
        "results": results,
        "threshold": thr,
        "model_name": p.model_name,
        "n": len(results),
        "fallback_warnings": unique[:25],
    }


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": API_TITLE, "version": API_VERSION, "docs": "/docs"}
