"""FastAPI application exposing the 30-day readmission model.

Endpoints
    GET  /health          liveness + which model is loaded
    GET  /metadata        full model card (metrics, features, split)
    GET  /schema/options  categorical option lists, for populating UI dropdowns
    POST /predict         one patient  -> probability, tier, SHAP drivers
    POST /predict/batch   many patients (<=500)

Interactive docs at /docs; the OpenAPI JSON at /openapi.json can be handed
straight to a frontend generator.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import API_TITLE, API_VERSION, REQUIRED_FIELDS, SERVER_COMPUTED
from .predictor import get_predictor
from .schemas import (
    BatchRequest, BatchResponse, HealthResponse,
    PatientFeatures, PredictionResponse,
)

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Predicts 30-day all-cause hospital readmission risk for Medicare patients "
        "from structured EHR data. Trained on the MIMIC-IV v3.1 Medicare cohort "
        "(244,576 admissions) using 67 features selected by recursive feature "
        "elimination.\n\n"
        "**Research prototype — not a medical device.** Outputs are intended to help "
        "prioritise transitional-care resources, never to withhold or reduce care. "
        "The model has not been externally validated or prospectively evaluated."
    ),
)

# Lovable previews get a fresh subdomain per deploy, so allow an explicit list via
# ALLOWED_ORIGINS and fall back to a permissive regex for *.lovable.app / localhost.
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_origin_regex=r"https://.*\.(lovable\.app|lovableproject\.com)|http://localhost:\d+",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    p = get_predictor()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=p.metadata.get("feature_set", "RFE-67"),
        n_features=len(p.feature_order),
        test_auroc=p.metadata["metrics"]["test_auroc"],
    )


@app.get("/metadata", tags=["ops"])
def metadata() -> dict:
    """Model card: metrics, feature list, training split, risk-tier cut-points."""
    p = get_predictor()
    return {
        **p.metadata,
        "required_fields": REQUIRED_FIELDS,
        "server_computed_features": sorted(SERVER_COMPUTED),
        "disclaimer": (
            "Research prototype trained on MIMIC-IV v3.1. Not externally validated, "
            "not prospectively evaluated, and not a medical device."
        ),
    }


@app.get("/schema/options", tags=["ops"])
def schema_options() -> dict:
    """Valid values for each categorical input, ordered by frequency in training.

    Use these to populate dropdowns so the UI cannot send an unseen category.
    """
    p = get_predictor()
    out: dict[str, list[str]] = {}
    for col, mapping in p.artifacts["cat_maps"].items():
        out[col] = sorted(mapping.keys())
    # race is a derivation-only input, so it is not in cat_maps
    out["race"] = sorted(p.artifacts["te_maps"]["race_te"].keys())
    return {"options": out, "counts": {k: len(v) for k, v in out.items()}}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(patient: PatientFeatures) -> PredictionResponse:
    try:
        result = get_predictor().predict(patient.model_dump())
    except Exception as exc:  # pragma: no cover - surfaced to the caller
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
def predict_batch(payload: BatchRequest) -> BatchResponse:
    predictor = get_predictor()
    try:
        results = [predictor.predict(p.model_dump()) for p in payload.patients]
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc
    return BatchResponse(count=len(results), predictions=[PredictionResponse(**r) for r in results])


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": API_TITLE, "version": API_VERSION, "docs": "/docs"}
