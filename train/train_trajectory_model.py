"""Train the survival:aft trajectory model on the SAME 67-feature matrix the
deployed binary model uses.

Feature set  : identical to the deployed model — the RFE-67 columns, encoded with
               the cat_maps persisted in artifacts/serving_artifacts.joblib, so the
               matrix here is bit-for-bit the one train_deployment_model.py built.
Labels       : time-to-readmission per the survival analysis reference
               (medicare-30day-readmission-mimic-iv/src/_run_daily_tdauc.py):
               look-ahead next_admittime per subject_id, event=1 only for
               readmission within 30 days whose next stay did not end in death,
               in-hospital deaths (time_to_event == 0) dropped, event-aware
               whole-day discretization (events ceil-clipped to [1, 29],
               censored = 30).
Split        : the exact patient-grouped split used throughout the paper.
Artifacts out: artifacts/aft_model.json, artifacts/aft_meta.json

Run:  <capstone-python> train/train_trajectory_model.py
"""
import os
N_JOBS = 8
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(N_JOBS)

import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parent
PUB = API_ROOT.parent                      # Publication Hospital Research
RESULTS = PUB / "medicare-30day-readmission-mimic-iv" / "results"
DATA = PUB / "Dataset" / "mimic-parquet"
ART = API_ROOT / "artifacts"
HORIZON_DAYS, SEED = 30, 42
np.random.seed(SEED)
t0 = time.time()

# ------------------------------------------------------- 1. feature matrix
# Same frame construction as train_deployment_model.py, same RFE-67 columns,
# and the *persisted* serving cat_maps so the encoding cannot drift from the API.
V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
V10 = pd.read_parquet(PUB / "training_table_v10.parquet")
assert V7["hadm_id"].equals(V10["hadm_id"]), "tables must be row-aligned"
v7_unique = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7_unique].reset_index(drop=True)], axis=1)

serving = joblib.load(ART / "serving_artifacts.joblib")
FEATURES = json.loads((RESULTS / "rfe_selection_results.json").read_text())["rfe_selected"]
assert len(FEATURES) == 67, f"expected 67 features, got {len(FEATURES)}"
assert FEATURES == serving["feature_order"], "feature order drifted from the deployed model"
CAT_FEATURES = serving["categorical_features"]
cat_maps = serving["cat_maps"]

X = DF[FEATURES].copy()
for c in CAT_FEATURES:
    X[c] = X[c].astype(str).fillna("__NA__").map(cat_maps[c]).astype(float)
X = X.apply(pd.to_numeric, errors="coerce")
Xv = X.to_numpy(dtype=np.float32)
print(f"[{time.time()-t0:.0f}s] matrix {Xv.shape}, categorical: {CAT_FEATURES}", flush=True)

# --------------------------------------------------- 2. time-to-event labels
# Exact copy of the cohort/event-time logic in _run_daily_tdauc.py.
ADM = pd.read_parquet(DATA / "admissions.parquet",
                      columns=["subject_id", "hadm_id", "admittime", "dischtime",
                               "deathtime", "hospital_expire_flag", "discharge_location"])
ADM = ADM.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
ADM["next_admittime"] = ADM.groupby("subject_id")["admittime"].shift(-1)
ADM["next_died"] = ADM.groupby("subject_id")["hospital_expire_flag"].shift(-1)

lab = V7.merge(ADM[["hadm_id", "next_admittime", "next_died", "hospital_expire_flag",
                    "discharge_location"]]
               .rename(columns={"discharge_location": "discharge_location_raw"}),
               on="hadm_id", how="left")
delta = (lab["next_admittime"] - lab["dischtime_dt"]).dt.total_seconds() / 86400.0
event = np.zeros(len(lab), dtype=np.int8)
ttime = np.full(len(lab), HORIZON_DAYS, dtype=np.float64)
in_hosp_death = (lab["hospital_expire_flag"] == 1) | (lab["discharge_location_raw"] == "DIED")
event[in_hosp_death] = 2; ttime[in_hosp_death] = 0.0
m_re = (~in_hosp_death) & delta.notna() & (delta > 0) & (delta <= HORIZON_DAYS) & (lab["next_died"].fillna(0) == 0)
event[m_re.values] = 1; ttime[m_re.values] = delta[m_re].values
m_cd = (~in_hosp_death) & delta.notna() & (delta > 0) & (delta <= HORIZON_DAYS) & (lab["next_died"].fillna(0) == 1)
event[m_cd.values] = 2; ttime[m_cd.values] = delta[m_cd].values
lab["event_type"] = event; lab["time_to_event"] = ttime
print(f"[{time.time()-t0:.0f}s] cohort built:",
      pd.Series(event).value_counts().sort_index().to_dict(), flush=True)

split = np.load(RESULTS / "v7_split_indices.npz")
train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]


def mask_alive(idx):
    """Drop in-hospital deaths (time_to_event == 0): no post-discharge window exists."""
    return idx[(lab.iloc[idx]["time_to_event"] > 0).values]


tr, va, te = mask_alive(train_idx), mask_alive(val_idx), mask_alive(test_idx)
print(f"[{time.time()-t0:.0f}s] alive-discharge split: {len(tr):,}/{len(va):,}/{len(te):,}", flush=True)


def yarr(idx):
    sub = lab.iloc[idx]
    e = (sub["event_type"] == 1).to_numpy()          # readmission is the event of interest
    t = sub["time_to_event"].to_numpy(dtype=float)
    # event-aware whole-day discretization (events capped 29, censoring 30)
    t = np.where(e, np.clip(np.ceil(t), 1, HORIZON_DAYS - 1), HORIZON_DAYS)
    return e, t


def dmat(idx):
    e, t = yarr(idx)
    dm = xgb.DMatrix(Xv[idx], feature_names=FEATURES)
    dm.set_float_info("label_lower_bound", t)
    dm.set_float_info("label_upper_bound", np.where(e, t, np.inf))
    return dm


dtr, dva = dmat(tr), dmat(va)

# ------------------------------------------------------------------ 3. train
params = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
          "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.0,
          "tree_method": "hist", "learning_rate": 0.05, "max_depth": 5,
          "subsample": 0.9, "colsample_bytree": 0.9, "nthread": N_JOBS, "seed": SEED}
bst = xgb.train(params, dtr, num_boost_round=600, evals=[(dva, "val")],
                early_stopping_rounds=50, verbose_eval=50)
best_it = int(bst.best_iteration)
print(f"[{time.time()-t0:.0f}s] trained; best_iteration={best_it}", flush=True)

# ------------------------------------------------ 4. test-set Harrell C-index
pred_time = bst.predict(xgb.DMatrix(Xv[te], feature_names=FEATURES))
risk = -pred_time                                   # shorter predicted time = riskier
e_te, t_te = yarr(te)

rng = np.random.default_rng(SEED)
sub = rng.choice(len(te), size=min(5000, len(te)), replace=False)
e_s, t_s, r_s = e_te[sub], t_te[sub], risk[sub]


def harrell_c(time_a, event_a, risk_a):
    """Pairwise Harrell C: comparable pairs are (i earlier event, j later or
    censored-at-the-same-time); ties in risk score count 0.5."""
    num = den = 0.0
    for i in np.where(event_a)[0]:
        cmp_mask = (time_a > time_a[i]) | ((time_a == time_a[i]) & ~event_a)
        den += cmp_mask.sum()
        num += (risk_a[i] > risk_a[cmp_mask]).sum() + 0.5 * (risk_a[i] == risk_a[cmp_mask]).sum()
    return float(num / den)


c_index = harrell_c(t_s, e_s, r_s)
print(f"[{time.time()-t0:.0f}s] test Harrell C (5,000-row subsample) = {c_index:.4f}", flush=True)

# ---------------------------------------------------------------- 5. persist
bst.save_model(str(ART / "aft_model.json"))
(ART / "aft_meta.json").write_text(json.dumps({
    "sigma": 1.0,
    "distribution": "normal",
    "harrell_c_test": round(c_index, 4),
    "trained": (f"{len(tr):,} train / {len(va):,} val / {len(te):,} test "
                "alive-discharge rows, patient-grouped v7 split"),
    "best_iteration": best_it,
}, indent=2))
print(f"[{time.time()-t0:.0f}s] wrote {ART / 'aft_model.json'} and {ART / 'aft_meta.json'}", flush=True)
