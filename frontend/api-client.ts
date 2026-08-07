/**
 * Typed client for the 30-Day Readmission Risk API.
 * Drop this into your Lovable project at src/lib/readmissionApi.ts
 * and set VITE_API_URL to your deployed backend.
 */

export const API_BASE =
  (import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000";

/* ------------------------------------------------------------------ types */

/** The 12 fields the API requires. */
export interface RequiredPatientFields {
  age_at_admit: number;
  los_days: number;
  admission_type: string;
  admission_location: string;
  discharge_location: string;
  /** ICD-10 chapter letter ("I", "J") or legacy 3-letter code ("cir", "rsp"). */
  primary_dx_chapter: string;
  /** Zero-padded 3-digit DRG, e.g. "291". */
  drg_code: string;
  n_diagnoses: number;
  prior_admissions_6m: number;
  prior_admissions_all: number;
  prior_readmission_count: number;
  /** Days since the previous discharge; send 365 when there is no prior stay. */
  time_since_last_discharge: number;
}

/** Everything else. Omit or send null and the server uses the training median. */
export interface OptionalPatientFields {
  race?: string | null;
  last_drg_dispo?: string | null;
  creatinine_last?: number | null;
  bun_last?: number | null;
  sodium_last?: number | null;
  hemoglobin_last?: number | null;
  albumin_last?: number | null;
  wbc_last?: number | null;
  glucose_last?: number | null;
  bicarbonate_last?: number | null;
  bilirubin_max?: number | null;
  lab_abnormal_rate?: number | null;
  n_labs_total?: number | null;
  n_lab_item_types?: number | null;
  n_lab_orders?: number | null;
  bmi_last?: number | null;
  bp_diastolic_outpatient?: number | null;
  elix_mets?: number | null;
  elix_solid_tumor?: number | null;
  elix_psychoses?: number | null;
  n_meds_total?: number | null;
  n_discharge_drugs?: number | null;
  distinct_drugs?: number | null;
  new_med_rate_48h?: number | null;
  n_emar_details?: number | null;
  iv_admin_rate?: number | null;
  med_orders_ratio?: number | null;
  discharge_hour?: number | null;
  n_procedures?: number | null;
  n_order_types?: number | null;
  orders_per_day?: number | null;
  orders_last_6h?: number | null;
  n_late_orders?: number | null;
  late_order_rate?: number | null;
  late_order_burden?: number | null;
  n_distinct_providers?: number | null;
  prior_mean_los_6m?: number | null;
  los_trend_180d?: number | null;
  freq_x_recency?: number | null;
  severity_composite?: number | null;
  clinical_complexity?: number | null;
  comorbidity_pc4?: number | null;
}

export type PatientFeatures = RequiredPatientFields & OptionalPatientFields;

export type RiskTier = "Low" | "Moderate" | "High" | "Very High";

export interface Driver {
  feature: string;
  /** Human-readable name, safe to render directly. */
  label: string;
  value: number;
  /** SHAP value on the log-odds scale. Positive pushes risk up. */
  contribution: number;
  direction: "increases" | "decreases";
  /** True when this value was assumed from population medians, not supplied.
   *  Mark these visually — never present an assumption as an observed finding. */
  imputed: boolean;
}

export interface PredictionResponse {
  readmission_probability: number;
  risk_tier: RiskTier;
  flagged: boolean;
  threshold: number;
  base_rate_log_odds: number;
  top_drivers: Driver[];
  /** Fields the server filled with the training median. Surface these to the user. */
  imputed_fields: string[];
  n_features_used: number;
  model_version: string;
}

export interface SchemaOptions {
  options: Record<string, string[]>;
  counts: Record<string, number>;
}

export interface HealthResponse {
  status: "ok";
  model_loaded: boolean;
  model_version: string;
  n_features: number;
  test_auroc: number;
}

/* ---------------------------------------------------------------- helpers */

class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly detail?: unknown) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!res.ok) {
    let detail: unknown;
    try {
      detail = (await res.json())?.detail;
    } catch {
      detail = await res.text();
    }
    // FastAPI returns 422 with a list of per-field validation errors
    if (res.status === 422 && Array.isArray(detail)) {
      const fields = detail
        .map((d: any) => `${d.loc?.slice(-1)[0]}: ${d.msg}`)
        .join("; ");
      throw new ApiError(`Validation failed — ${fields}`, 422, detail);
    }
    throw new ApiError(
      typeof detail === "string" ? detail : `Request failed (${res.status})`,
      res.status,
      detail,
    );
  }
  return res.json() as Promise<T>;
}

/* ------------------------------------------------------------------- api */

export const readmissionApi = {
  health: () => request<HealthResponse>("/health"),

  metadata: () => request<Record<string, unknown>>("/metadata"),

  /** Fetch once on mount to populate every dropdown. */
  options: () => request<SchemaOptions>("/schema/options"),

  predict: (patient: PatientFeatures) =>
    request<PredictionResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(patient),
    }),

  predictBatch: (patients: PatientFeatures[]) =>
    request<{ count: number; predictions: PredictionResponse[] }>(
      "/predict/batch",
      { method: "POST", body: JSON.stringify({ patients }) },
    ),
};

/* ---------------------------------------------------------- presentation */

export const TIER_STYLES: Record<RiskTier, { label: string; className: string }> = {
  Low: { label: "Low risk", className: "bg-emerald-100 text-emerald-800 border-emerald-200" },
  Moderate: { label: "Moderate risk", className: "bg-amber-100 text-amber-800 border-amber-200" },
  High: { label: "High risk", className: "bg-orange-100 text-orange-800 border-orange-200" },
  "Very High": { label: "Very high risk", className: "bg-red-100 text-red-800 border-red-200" },
};

export const formatProbability = (p: number) => `${(p * 100).toFixed(1)}%`;

/**
 * Normalise SHAP contributions to 0-100 bar widths, relative to the largest
 * absolute contribution in the set.
 */
export function driverBarWidths(drivers: Driver[]): number[] {
  const max = Math.max(...drivers.map((d) => Math.abs(d.contribution)), 1e-9);
  return drivers.map((d) => (Math.abs(d.contribution) / max) * 100);
}
