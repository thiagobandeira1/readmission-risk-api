"""End-to-end API tests against the RiskPath Console contract.

Run:  <capstone-python> -m pytest tests -q
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)

MINIMAL = {
    "age_at_admit": 76, "los_days": 5.4, "admission_type": "EW EMER.",
    "admission_location": "EMERGENCY ROOM", "discharge_location": "HOME HEALTH CARE",
    "primary_dx_chapter": "I", "drg_code": "291", "n_diagnoses": 12,
    "prior_admissions_6m": 1, "prior_admissions_all": 3,
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


# ------------------------------------------------------------------ metadata
def test_health():
    b = client.get("/health").json()
    assert b["status"] == "ok"
    assert b["model"].startswith("xgboost-rfe67")


def test_metadata_shape_matches_frontend_contract():
    b = client.get("/metadata").json()
    assert isinstance(b["features"], list) and b["features"]
    assert 0.0 < b["default_threshold"] < 1.0
    mi = b["model_info"]
    for key in ("name", "seed", "n_features", "published_test_auroc", "deployed_test_auroc"):
        assert key in mi
    assert mi["n_features"] == 67
    for f in b["features"]:
        assert f["type"] in {"numeric", "categorical"}
        assert "pct_nan" in f
        if f["type"] == "numeric":
            assert f["min"] <= f["median"] <= f["max"]
        else:
            assert f["levels"]


def test_metadata_never_exposes_server_derived_features():
    names = {f["name"] for f in client.get("/metadata").json()["features"]}
    for derived in ("bun_creatinine_ratio", "discharge_location_te", "n_meds_x_age",
                    "log_time_since_discharge", "race_te"):
        assert derived not in names, f"{derived} must be derived, not requested"


def test_examples_are_predictable():
    b = client.get("/examples?n=5").json()
    assert b["n"] == len(b["examples"]) == 5
    for ex in b["examples"]:
        assert client.post("/predictions", json=ex).status_code == 200


# ----------------------------------------------------------------- inference
def test_predictions_contract():
    r = client.post("/predictions?threshold=0.2", json=MINIMAL)
    assert r.status_code == 200, r.text
    b = r.json()
    assert 0.0 <= b["probability"] <= 1.0
    assert b["prediction"] in (0, 1)
    assert b["threshold"] == 0.2
    assert b["prediction"] == int(b["probability"] >= 0.2)
    assert isinstance(b["fallback_warnings"], list)


def test_threshold_defaults_to_model_operating_point():
    default = client.get("/metadata").json()["default_threshold"]
    assert client.post("/predictions", json=MINIMAL).json()["threshold"] == default


def test_explanations_align_with_the_model_vector():
    b = client.post("/explanations", json=RICH).json()
    n = len(b["feature_names"])
    assert n == 67
    assert len(b["shap_values"]) == n
    assert len(b["feature_values_transformed"]) == n
    # SHAP values plus the base term must reconstruct the predicted log-odds
    import math
    logit = b["base_value"] + sum(b["shap_values"])
    assert math.isclose(1 / (1 + math.exp(-logit)), b["probability"], abs_tol=2e-3)


def test_richer_payload_produces_fewer_warnings():
    lean = client.post("/predictions", json=MINIMAL).json()["fallback_warnings"]
    rich = client.post("/predictions", json=RICH).json()["fallback_warnings"]
    assert len(rich) < len(lean)


def test_prediction_is_deterministic():
    a = client.post("/predictions", json=RICH).json()["probability"]
    b = client.post("/predictions", json=RICH).json()["probability"]
    assert a == b


def test_risk_ordering_is_clinically_sensible():
    high = {**RICH, "prior_admissions_6m": 6, "prior_admissions_all": 15,
            "prior_readmission_count": 5, "time_since_last_discharge": 3}
    low = {**RICH, "prior_admissions_6m": 0, "prior_admissions_all": 0,
           "prior_readmission_count": 0, "time_since_last_discharge": 365,
           "admission_type": "ELECTIVE", "discharge_location": "HOME"}
    assert (client.post("/predictions", json=high).json()["probability"]
            > client.post("/predictions", json=low).json()["probability"])


def test_batch_matches_single_predictions():
    b = client.post("/predictions/batch?threshold=0.25",
                    json={"patients": [MINIMAL, RICH]}).json()
    assert b["n"] == 2 and b["threshold"] == 0.25
    single = client.post("/predictions?threshold=0.25", json=MINIMAL).json()["probability"]
    assert b["results"][0]["probability"] == single


def test_batch_accepts_wrapped_rows():
    b = client.post("/predictions/batch", json={"patients": [{"features": MINIMAL}]}).json()
    assert b["n"] == 1


# ------------------------------------------------------------------ resilience
def test_unseen_category_falls_back_without_crashing():
    r = client.post("/predictions", json={**MINIMAL, "drg_code": "NOT_A_REAL_DRG"})
    assert r.status_code == 200


def test_missing_optional_fields_are_imputed_not_rejected():
    r = client.post("/predictions", json={"age_at_admit": 70, "los_days": 3})
    assert r.status_code == 200
    assert r.json()["fallback_warnings"]


# ------------------------------------------------------------------ trajectory
def test_metadata_advertises_trajectory():
    assert client.get("/metadata").json()["trajectory_available"] is True


def test_trajectory_is_a_valid_cumulative_curve():
    b = client.post("/predictions/trajectory", json=RICH).json()
    days, cum = b["days"], b["cumulative_probability"]
    assert days == list(range(1, 31))
    assert len(cum) == len(b["daily_increment"]) == 30
    assert all(0.0 <= v <= 1.0 for v in cum)
    assert all(cum[i] >= cum[i - 1] for i in range(1, 30)), "cumulative must be non-decreasing"
    assert b["horizon_probability"] == cum[-1]
    assert b["model_name"].startswith("xgboost-aft")
    assert "research prototype" in b["disclaimer"]


def test_daily_increments_reconstruct_the_curve():
    b = client.post("/predictions/trajectory", json=RICH).json()
    running = 0.0
    for inc, cum in zip(b["daily_increment"], b["cumulative_probability"]):
        running += inc
        assert abs(running - cum) < 1e-4


def test_trajectory_orders_high_and_low_risk_patients():
    high = {**RICH, "prior_admissions_6m": 6, "prior_admissions_all": 15,
            "prior_readmission_count": 5, "time_since_last_discharge": 3,
            "discharge_location": "SKILLED NURSING FACILITY"}
    low = {**RICH, "prior_admissions_6m": 0, "prior_admissions_all": 0,
           "prior_readmission_count": 0, "time_since_last_discharge": 365,
           "admission_type": "ELECTIVE", "discharge_location": "HOME"}
    h = client.post("/predictions/trajectory", json=high).json()
    lo = client.post("/predictions/trajectory", json=low).json()
    assert h["cumulative_probability"][6] > lo["cumulative_probability"][6]   # day 7
    assert h["horizon_probability"] > lo["horizon_probability"]
    # a higher-risk patient is expected back sooner
    assert h["median_predicted_day"] < lo["median_predicted_day"]


def test_trajectory_imputes_missing_fields():
    b = client.post("/predictions/trajectory", json={"age_at_admit": 70, "los_days": 3}).json()
    assert len(b["cumulative_probability"]) == 30
    assert b["fallback_warnings"]


def test_errors_use_the_frontend_envelope():
    r = client.post("/predictions/batch", json={"patients": "not-a-list"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"
