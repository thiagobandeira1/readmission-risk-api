"""Request/response contracts.

Only raw clinical and operational values are accepted. Everything the model needs
beyond these (interactions, log transforms, target encodings) is derived by the
server, so the client never has to know the model's internal feature engineering.
Optional fields left null fall back to the training-set median and are echoed back
in `imputed_fields`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    # ---------------- required: admission context ----------------
    age_at_admit: float = Field(..., ge=18, le=120, description="Age in years at admission", examples=[76])
    los_days: float = Field(..., ge=0, le=400, description="Length of stay in days", examples=[5.4])
    admission_type: str = Field(..., description="e.g. EW EMER., URGENT, ELECTIVE", examples=["EW EMER."])
    admission_location: str = Field(..., description="e.g. EMERGENCY ROOM, PHYSICIAN REFERRAL", examples=["EMERGENCY ROOM"])
    discharge_location: str = Field(..., description="e.g. HOME, HOME HEALTH CARE, SKILLED NURSING FACILITY", examples=["HOME HEALTH CARE"])
    primary_dx_chapter: str = Field(..., description="ICD chapter of the primary diagnosis. ICD-10 single letter (e.g. 'I' = circulatory, 'J' = respiratory) or the legacy 3-letter code (e.g. 'cir', 'rsp'). Call /schema/options for the full list.", examples=["I"])
    drg_code: str = Field(..., description="Diagnosis-Related Group code, zero-padded 3 digits", examples=["291"])
    n_diagnoses: float = Field(..., ge=0, le=100, description="Number of coded diagnoses", examples=[12])

    # ---------------- required: prior utilisation ----------------
    prior_admissions_6m: float = Field(..., ge=0, le=100, description="Admissions in the previous 6 months", examples=[1])
    prior_admissions_all: float = Field(..., ge=0, le=500, description="All prior admissions on record", examples=[3])
    prior_readmission_count: float = Field(..., ge=0, le=200, description="Prior 30-day readmissions", examples=[1])
    time_since_last_discharge: float = Field(..., ge=0, le=100000, description="Days since the previous discharge (use 365 if none)", examples=[45])

    # ---------------- optional: demographics / derivation inputs ----------------
    race: str | None = Field(None, description="Self-reported race; used only for the risk-encoding lookup", examples=["WHITE"])
    last_drg_dispo: str | None = Field(None, description="DRG description/disposition text from the prior stay")

    # ---------------- optional: laboratory values ----------------
    creatinine_last: float | None = Field(None, ge=0, le=30, description="Creatinine, last value (mg/dL)")
    bun_last: float | None = Field(None, ge=0, le=300, description="Blood urea nitrogen, last value (mg/dL)")
    sodium_last: float | None = Field(None, ge=80, le=200, description="Sodium, last value (mEq/L)")
    hemoglobin_last: float | None = Field(None, ge=0, le=30, description="Hemoglobin, last value (g/dL)")
    albumin_last: float | None = Field(None, ge=0, le=10, description="Albumin, last value (g/dL)")
    wbc_last: float | None = Field(None, ge=0, le=200, description="White blood cell count, last value (K/uL)")
    glucose_last: float | None = Field(None, ge=0, le=2000, description="Glucose, last value (mg/dL)")
    bicarbonate_last: float | None = Field(None, ge=0, le=60, description="Bicarbonate, last value (mEq/L)")
    bilirubin_max: float | None = Field(None, ge=0, le=80, description="Bilirubin, maximum during stay (mg/dL)")
    lab_abnormal_rate: float | None = Field(None, ge=0, le=1, description="Fraction of labs flagged abnormal")
    n_labs_total: float | None = Field(None, ge=0, description="Total lab results during the stay")
    n_lab_item_types: float | None = Field(None, ge=0, description="Distinct lab item types ordered")
    n_lab_orders: float | None = Field(None, ge=0, description="Lab orders placed")

    # ---------------- optional: vitals / measurements ----------------
    bmi_last: float | None = Field(None, ge=5, le=100, description="Body-mass index, last recorded")
    bp_diastolic_outpatient: float | None = Field(None, ge=20, le=200, description="Outpatient diastolic blood pressure")

    # ---------------- optional: comorbidity flags ----------------
    elix_mets: float | None = Field(None, ge=0, le=1, description="Metastatic cancer (0/1)")
    elix_solid_tumor: float | None = Field(None, ge=0, le=1, description="Solid tumour without metastasis (0/1)")
    elix_psychoses: float | None = Field(None, ge=0, le=1, description="Psychoses (0/1)")

    # ---------------- optional: medications ----------------
    n_meds_total: float | None = Field(None, ge=0, description="Total medication orders")
    n_discharge_drugs: float | None = Field(None, ge=0, description="Medications on the discharge list")
    distinct_drugs: float | None = Field(None, ge=0, description="Distinct drugs administered")
    new_med_rate_48h: float | None = Field(None, ge=0, description="Rate of new medications in the first 48 hours")
    n_emar_details: float | None = Field(None, ge=0, description="eMAR administration records")
    iv_admin_rate: float | None = Field(None, ge=0, description="IV administrations per day")
    med_orders_ratio: float | None = Field(None, ge=0, le=1, description="Medication orders as a share of all orders")

    # ---------------- optional: operational / order activity ----------------
    discharge_hour: float | None = Field(None, ge=0, le=23, description="Hour of day of discharge (0-23)")
    n_procedures: float | None = Field(None, ge=0, description="Procedures performed")
    n_order_types: float | None = Field(None, ge=0, description="Distinct order types")
    orders_per_day: float | None = Field(None, ge=0, description="Provider orders per day")
    orders_last_6h: float | None = Field(None, ge=0, description="Orders in the final 6 hours before discharge")
    n_late_orders: float | None = Field(None, ge=0, description="Orders placed late in the stay")
    late_order_rate: float | None = Field(None, ge=0, le=1, description="Late orders as a share of all orders")
    late_order_burden: float | None = Field(None, ge=0, description="Composite late-order burden")
    n_distinct_providers: float | None = Field(None, ge=0, description="Distinct providers involved in care")

    # ---------------- optional: prior-stay dynamics ----------------
    prior_mean_los_6m: float | None = Field(None, ge=0, description="Mean length of stay across prior 6-month admissions")
    los_trend_180d: float | None = Field(None, description="Trend in length of stay over the prior 180 days")
    freq_x_recency: float | None = Field(None, ge=0, description="Admission frequency weighted by recency")

    # ---------------- optional: composites ----------------
    severity_composite: float | None = Field(None, ge=0, description="Composite severity score")
    clinical_complexity: float | None = Field(None, ge=0, description="Composite clinical-complexity score")
    comorbidity_pc4: float | None = Field(None, description="4th principal component of the comorbidity profile")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "age_at_admit": 76, "los_days": 5.4, "admission_type": "EW EMER.",
                "admission_location": "EMERGENCY ROOM", "discharge_location": "HOME HEALTH CARE",
                "primary_dx_chapter": "I", "drg_code": "291",
                "n_diagnoses": 12, "prior_admissions_6m": 1, "prior_admissions_all": 3,
                "prior_readmission_count": 1, "time_since_last_discharge": 45,
                "race": "WHITE", "creatinine_last": 1.4, "bun_last": 28, "sodium_last": 136,
                "hemoglobin_last": 9.8, "albumin_last": 3.1, "n_discharge_drugs": 14,
            }]
        }
    }


class Driver(BaseModel):
    feature: str
    label: str
    value: float
    contribution: float = Field(..., description="SHAP contribution on the log-odds scale")
    direction: Literal["increases", "decreases"]
    imputed: bool = Field(..., description="True if this value was filled from the training median rather than supplied by the caller")


class PredictionResponse(BaseModel):
    readmission_probability: float = Field(..., description="Calibrated probability of readmission within 30 days")
    risk_tier: Literal["Low", "Moderate", "High", "Very High"]
    flagged: bool = Field(..., description="True when the probability meets the operating threshold")
    threshold: float
    base_rate_log_odds: float
    top_drivers: list[Driver] = Field(..., description="Largest SHAP contributions for this patient")
    imputed_fields: list[str] = Field(..., description="Fields filled with the training median because none was supplied")
    n_features_used: int
    model_version: str


class BatchRequest(BaseModel):
    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str
    n_features: int
    test_auroc: float
