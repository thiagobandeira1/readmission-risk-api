"""End-to-end API tests. Run:  <capstone-python> -m pytest tests -q"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

MINIMAL = {
    "age_at_admit": 76, "los_days": 5.4, "admission_type": "EW EMER.",
    "admission_location": "EMERGENCY ROOM", "discharge_location": "HOME HEALTH CARE",
    "primary_dx_chapter": "I", "drg_code": "291",
    "n_diagnoses": 12, "prior_admissions_6m": 1, "prior_admissions_all": 3,
    "prior_readmission_count": 1, "time_since_last_discharge": 45,
}
RICH = {
    **MINIMAL, "race": "WHITE", "creatinine_last": 1.4, "bun_last": 28,
    "sodium_last": 136, "hemoglobin_last": 9.8, "albumin_last": 3.1,
    "wbc_last": 8.2, "glucose_last": 142, "bicarbonate_last": 24,
    "bilirubin_max": 1.1, "n_discharge_drugs": 14, "n_meds_total": 40,
    "los_trend_180d": 1.2, "severity_composite": 2.0, "lab_abnormal_rate": 0.35,
    "n_distinct_providers": 9, "bmi_last": 28.4,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["n_features"] == 67
    assert body["test_auroc"] > 0.75


def test_metadata_lists_contract():
    body = client.get("/metadata").json()
    assert body["n_features"] == 67
    assert "time_since_last_discharge" in body["required_fields"]
    # derived features must never be requested from the client
    assert "bun_creatinine_ratio" in body["server_computed_features"]


def test_options_populates_dropdowns():
    body = client.get("/schema/options").json()
    for key in ("discharge_location", "admission_type", "primary_dx_chapter", "race"):
        assert body["options"][key], f"{key} should have options"


def test_predict_minimal_payload():
    r = client.post("/predict", json=MINIMAL)
    assert r.status_code == 200, r.text
    b = r.json()
    assert 0.0 <= b["readmission_probability"] <= 1.0
    assert b["risk_tier"] in {"Low", "Moderate", "High", "Very High"}
    assert len(b["top_drivers"]) == 8
    # with only the required fields, optional ones must be reported as imputed
    assert b["imputed_fields"], "expected imputation to be disclosed"


def test_richer_payload_imputes_less():
    lean = client.post("/predict", json=MINIMAL).json()
    rich = client.post("/predict", json=RICH).json()
    assert len(rich["imputed_fields"]) < len(lean["imputed_fields"])


def test_prediction_is_deterministic():
    a = client.post("/predict", json=RICH).json()
    b = client.post("/predict", json=RICH).json()
    assert a["readmission_probability"] == b["readmission_probability"]


def test_risk_ordering_is_sensible():
    """A frequently-readmitted patient should outrank a first-time elective one."""
    high = {**RICH, "prior_admissions_6m": 6, "prior_admissions_all": 15,
            "prior_readmission_count": 5, "time_since_last_discharge": 3}
    low = {**RICH, "prior_admissions_6m": 0, "prior_admissions_all": 0,
           "prior_readmission_count": 0, "time_since_last_discharge": 365,
           "admission_type": "ELECTIVE", "discharge_location": "HOME"}
    p_high = client.post("/predict", json=high).json()["readmission_probability"]
    p_low = client.post("/predict", json=low).json()["readmission_probability"]
    assert p_high > p_low


def test_unseen_category_does_not_crash():
    r = client.post("/predict", json={**MINIMAL, "drg_code": "NOT_A_REAL_DRG"})
    assert r.status_code == 200


def test_validation_rejects_impossible_age():
    r = client.post("/predict", json={**MINIMAL, "age_at_admit": 400})
    assert r.status_code == 422


def test_missing_required_field_rejected():
    payload = {k: v for k, v in MINIMAL.items() if k != "los_days"}
    assert client.post("/predict", json=payload).status_code == 422


def test_batch():
    r = client.post("/predict/batch", json={"patients": [MINIMAL, RICH]})
    assert r.status_code == 200
    assert r.json()["count"] == 2
