# Wiring the Lovable frontend to this API

Two steps: paste the prompt below into Lovable, then point it at your deployed backend.

---

## Step 1 — paste this into Lovable

> Update the app to call a real prediction API instead of any mock/placeholder logic.
>
> **API base URL:** read from `VITE_API_URL` (fall back to `http://localhost:8000`).
>
> ### Endpoints
> - `GET /schema/options` → `{ options: { admission_type: string[], admission_location: string[], discharge_location: string[], primary_dx_chapter: string[], drg_code: string[], last_drg_dispo: string[], race: string[] } }`
>   Call once on mount and use it to populate every dropdown. Never hard-code these lists.
> - `POST /predict` → send one patient object, receive the prediction.
> - `GET /health` → `{ status, model_version, n_features, test_auroc }` for a status badge.
>
> ### The form
>
> Build a single-page "Discharge Risk Assessment" form in three collapsible sections.
> Only the first section is required; the other two improve accuracy and start collapsed.
>
> **Section 1 — Admission & History (all required):**
> - `age_at_admit` — number, 18–120
> - `los_days` — number, 0–400, one decimal
> - `admission_type` — select from options
> - `admission_location` — select from options
> - `discharge_location` — select from options
> - `primary_dx_chapter` — select from options (ICD chapter codes)
> - `drg_code` — searchable select (there are ~390 values, so it needs a type-ahead)
> - `n_diagnoses` — number, 0–100
> - `prior_admissions_6m` — number, 0–100
> - `prior_admissions_all` — number, 0–500
> - `prior_readmission_count` — number, 0–200
> - `time_since_last_discharge` — number in days; add helper text "enter 365 if this is the patient's first admission"
>
> **Section 2 — Labs & Vitals (all optional):**
> `creatinine_last`, `bun_last`, `sodium_last`, `hemoglobin_last`, `albumin_last`,
> `wbc_last`, `glucose_last`, `bicarbonate_last`, `bilirubin_max`, `bmi_last`,
> `bp_diastolic_outpatient`, `lab_abnormal_rate` (0–1)
> Show the unit next to each label: mg/dL, mEq/L, g/dL, K/uL.
>
> **Section 3 — Medications, Orders & Comorbidities (all optional):**
> `n_meds_total`, `n_discharge_drugs`, `distinct_drugs`, `n_procedures`,
> `n_distinct_providers`, `orders_per_day`, `n_late_orders`, `discharge_hour` (0–23),
> `los_trend_180d`, `prior_mean_los_6m`, `severity_composite`, `clinical_complexity`,
> and toggles for `elix_mets`, `elix_solid_tumor`, `elix_psychoses` (send 1 or 0).
> Also a `race` select (used only for a risk-encoding lookup).
>
> **Important:** omit optional fields entirely, or send `null` — never send `0` for
> "unknown". Zero is a real clinical value and would corrupt the prediction.
>
> ### The result panel
>
> On success show:
> 1. **A large probability** — `readmission_probability` formatted as a percentage with
>    one decimal (e.g. "19.6%"), labelled "30-day readmission risk".
> 2. **A risk-tier badge** — `risk_tier` is `"Low" | "Moderate" | "High" | "Very High"`.
>    Colour them emerald / amber / orange / red respectively.
> 3. **A "flagged for follow-up" indicator** when `flagged` is true, with a tooltip
>    explaining it means the score met the model's operating threshold.
> 4. **A "Why this score" list** from `top_drivers` (8 items). For each, render a
>    horizontal diverging bar: the bar length is `|contribution|` normalised against the
>    largest absolute contribution in the list; red/right when `direction` is
>    `"increases"`, blue/left when `"decreases"`. Show `label` as the row title and
>    `value` as a muted subtitle. Add a footnote: *"Contributions are SHAP values showing
>    how each factor moved this patient's score. They explain the model, not causation."*
> 5. **An assumptions notice** — if `imputed_fields` is non-empty, show a collapsible
>    info box: "N optional fields were estimated from population averages. Supplying
>    them would improve accuracy." List the field names inside.
>
> ### Behaviour
> - Show a skeleton/spinner while the request is in flight.
> - On a 422, surface the per-field messages from `detail` next to the offending inputs.
> - On a network error, show a retry button and a note that the API may be starting up
>   (free hosting tiers cold-start and the first request can take ~30s).
> - Add a "Load example patient" button that fills the form with the values from the
>   `/openapi.json` example so the demo is one click.
>
> ### Required disclaimer
>
> Show this persistently in the footer, and in a dismissible banner on first load:
>
> > **Research prototype — not a medical device.** This model has not been externally
> > validated or prospectively evaluated. It is intended to help prioritise
> > transitional-care resources, never to withhold or reduce care. Do not use it to make
> > individual clinical decisions.
>
> ### Style
> Clean clinical dashboard: white background, generous whitespace, one accent colour
> (teal `#1F7A8C`), coral `#E76F51` for risk emphasis. Cards with soft shadows, no
> gradients. Fully responsive; the form should be usable on a tablet.

---

## Step 2 — connect it

1. In Lovable, add an environment variable **`VITE_API_URL`** = your deployed backend URL
   (e.g. `https://readmission-api.onrender.com`). No trailing slash.
2. Deploy the backend (see the repo README — Docker works on Render, Railway, Fly.io).
3. Set **`ALLOWED_ORIGINS`** on the backend to your Lovable app URL, so CORS is
   restricted to your frontend rather than the permissive default.
4. Confirm the wiring: the app's status badge should show `model_version: RFE-67` and
   `test_auroc: 0.7966`.

`api-client.ts` in this folder is a ready-made typed client — paste it into
`src/lib/readmissionApi.ts` if you would rather wire calls by hand than let Lovable
generate the fetch layer.

## Local development against the API

```bash
# terminal 1 — backend
uvicorn app.main:app --reload

# terminal 2 — frontend
VITE_API_URL=http://localhost:8000 npm run dev
```

CORS already allows `localhost` on any port, plus `*.lovable.app` and
`*.lovableproject.com`.
