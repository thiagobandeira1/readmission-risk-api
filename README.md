# 30-Day Readmission Risk API

FastAPI service that predicts 30-day all-cause hospital readmission risk for Medicare
patients from structured EHR data, and explains every prediction.

Backing research: *Predicting 30-Day Hospital Readmission in Medicare Patients — an
interpretable gradient-boosting model on MIMIC-IV v3.1.*

> **Research prototype, not a medical device.** The model has not been externally
> validated or prospectively evaluated. It is intended to help *allocate* transitional-care
> resources, never to withhold or reduce care.

## Model

| | |
|---|---|
| Algorithm | XGBoost (gradient-boosted trees) |
| Features | **67**, selected by recursive feature elimination from a 207-feature pool |
| Training data | MIMIC-IV v3.1 Medicare cohort — 244,576 admissions, 21.1% readmitted |
| Split | Patient-grouped on `subject_id` (no patient in both train and test) |
| **Test AUROC** | **0.7966** |
| Average precision | 0.5194 (vs 0.209 base rate) |
| Brier score | 0.1324 |
| **Calibration (ECE)** | **0.0055** — mean predicted 0.2099 vs observed 0.2091 |

Calibration matters here: the API returns a probability meant to be read literally,
so that a score can be compared against an intervention budget.

## Design: the client sends raw values, the server does the feature engineering

Of the 67 model features, 16 are not things anybody can type into a form — 11 are
algebraic derivations and 5 are target encodings that require maps fitted on the
training split. Asking a UI for `discharge_location_te = 0.2637` would be absurd and
would let clients drift away from how the model was trained.

So the contract is:

```
UI  →  raw clinical + operational values
          →  API derives interactions, log transforms, target encodings
          →  model.predict_proba
          →  { probability, risk_tier, top SHAP drivers, imputed_fields }
```

Each of the 11 derivations was verified against the training data and reproduces its
column with **zero error** across all 244,576 admissions (see `app/features.py`).

**12 required fields** (clinically essential):
`age_at_admit`, `los_days`, `admission_type`, `admission_location`, `discharge_location`,
`primary_dx_chapter`, `drg_code`, `n_diagnoses`, `prior_admissions_6m`,
`prior_admissions_all`, `prior_readmission_count`, `time_since_last_discharge`

Everything else is optional. Any optional field left `null` is filled with the
training-set median and named in the response's `imputed_fields`, so a caller always
knows what was assumed on their behalf. More fields supplied → fewer assumptions.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + model identifier |
| `GET` | `/metadata` | Feature schema (drives the UI form), model card, default threshold |
| `GET` | `/examples?n=5` | Synthetic demo patients |
| `POST` | `/predictions?threshold=` | One patient → probability + class |
| `POST` | `/explanations` | One patient → SHAP attributions over all 67 features |
| `POST` | `/predictions/batch?threshold=` | Up to 1000 patients |

Requests are a flat `{feature: value}` object. Errors use the envelope
`{"error": {"code", "message", "details"}}`, with `code` one of `VALIDATION_ERROR`
or `INTERNAL_ERROR`. Fields the caller omits are imputed and reported in
`fallback_warnings`.

`/metadata` deliberately lists **only** the 54 client-supplied fields — the 16 the
server derives are absent, so the UI can never be asked for a target encoding.

Interactive docs at `/docs`; machine-readable schema at `/openapi.json`.

### Example

```bash
curl -X POST http://localhost:8000/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "age_at_admit": 76, "los_days": 5.4,
    "admission_type": "EW EMER.", "admission_location": "EMERGENCY ROOM",
    "discharge_location": "HOME HEALTH CARE",
    "primary_dx_chapter": "I",
    "drg_code": "291", "n_diagnoses": 12,
    "prior_admissions_6m": 1, "prior_admissions_all": 3,
    "prior_readmission_count": 1, "time_since_last_discharge": 45,
    "race": "WHITE", "creatinine_last": 1.4, "bun_last": 28,
    "albumin_last": 3.1, "n_discharge_drugs": 14
  }'
```

```jsonc
{
  "probability": 0.195653,
  "prediction": 0,
  "threshold": 0.196578,
  "model_name": "xgboost-rfe67-seed42",
  "fallback_warnings": [
    "'bmi_last' was not supplied; using the training median (27.7)."
  ]
}
```

`POST /explanations` returns `shap_values`, `base_value`, `feature_names` and
`feature_values_transformed` for all 67 model features. SHAP values are on the
log-odds scale and explain what the *model* did — they are not causal claims about
the patient.

## Frontend

The **RiskPath Console** frontend is schema-driven: it renders its entire form from
`/metadata`, so adding or removing a model feature requires no frontend change.
See `frontend/LOVABLE_PROMPT.md`.

## Run locally

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs

```bash
pytest tests -q
```

## Deploy

Any Docker host works. The container reads `$PORT`, so Render, Railway, Fly.io and
Cloud Run all work unmodified.

```bash
docker build -t readmission-api .
docker run -p 8000:8000 readmission-api
```

Set `ALLOWED_ORIGINS` to your frontend's URL in production (comma-separated). Left
unset, CORS falls back to a regex allowing `*.lovable.app` and `localhost`.

## Retraining

```bash
python train/train_deployment_model.py
```

Rebuilds the model and every serving artifact (target-encoding maps, categorical
encoders, medians, operating threshold, risk-tier cut-points) into `artifacts/`.
Requires the MIMIC-IV derived tables, which are **not** in this repository —
MIMIC-IV is credentialed data available from PhysioNet under a data-use agreement.

## Layout

```
app/
  config.py      feature contract: what is client-supplied vs server-derived
  features.py    the 11 verified derivations + target encoding
  predictor.py   model loading, inference, TreeSHAP attributions
  schemas.py     Pydantic request/response models with clinical range validation
  main.py        FastAPI routes and CORS
artifacts/       model.json, serving_artifacts.joblib, metadata.json
train/           retraining script
tests/           end-to-end API tests
frontend/        TypeScript client + Lovable integration guide
```

## Limitations

- Single-centre training data (Beth Israel Deaconess); no external validation yet.
- Readmissions to *other* hospitals are invisible in MIMIC-IV, so some training
  negatives are truly positives.
- No social-determinant variables, which strongly influence readmission.
- Discrimination is lower for Black patients (AUROC 0.754) than White patients (0.791)
  in the source study; calibration is comparable across groups. Disclosed, not resolved.
