"""Train and persist the 67-feature deployment model + every artifact the API needs.

Feature set  : the 67 features selected by recursive feature elimination
Split        : the exact patient-grouped split used throughout the paper
Artifacts out: artifacts/model.json, artifacts/serving_artifacts.joblib, artifacts/metadata.json

Run:  <capstone-python> train/train_deployment_model.py
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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parent
PUB = API_ROOT.parent                      # Publication Hospital Research
RESULTS = PUB / "medicare-30day-readmission-mimic-iv" / "results"
DATA = PUB / "Dataset" / "mimic-parquet"
ART = API_ROOT / "artifacts"
ART.mkdir(exist_ok=True)
SEED = 42
t0 = time.time()

# ---------------------------------------------------------------- 1. data
V7 = pd.read_parquet(PUB / "training_table_v7.parquet")
V10 = pd.read_parquet(PUB / "training_table_v10.parquet")
assert V7["hadm_id"].equals(V10["hadm_id"]), "tables must be row-aligned"
v7_unique = [c for c in V7.columns if c not in V10.columns]
DF = pd.concat([V10.reset_index(drop=True), V7[v7_unique].reset_index(drop=True)], axis=1)

# race lives only in the raw admissions table; needed to rebuild the race_te map
race = pd.read_parquet(DATA / "admissions.parquet", columns=["hadm_id", "race"])
DF = DF.merge(race, on="hadm_id", how="left")
DF["race"] = DF["race"].fillna("UNKNOWN")

FEATURES = json.loads((RESULTS / "rfe_selection_results.json").read_text())["rfe_selected"]
assert len(FEATURES) == 67, f"expected 67 features, got {len(FEATURES)}"
y = DF["readmit_30d"].to_numpy().astype(int)
print(f"[{time.time()-t0:.0f}s] frame {DF.shape}, features {len(FEATURES)}, prevalence {y.mean():.4f}", flush=True)

split = np.load(RESULTS / "v7_split_indices.npz")
tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]
print(f"train/val/test: {len(tr):,}/{len(va):,}/{len(te):,}", flush=True)

# ---------------------------------------------------------------- 2. serving artifacts
# 2a. target-encoding maps: category -> mean of the encoded column over TRAIN rows only
TE_PAIRS = {
    "primary_dx_chapter_te": "primary_dx_chapter",
    "discharge_location_te": "discharge_location",
    "drg_code_te":           "drg_code",
    "last_drg_dispo_te":     "last_drg_dispo",
    "race_te":               "race",
}
te_maps, te_globals = {}, {}
train_df = DF.iloc[tr]
for te_col, raw_col in TE_PAIRS.items():
    g = train_df.groupby(train_df[raw_col].astype(str))[te_col].mean()
    te_maps[te_col] = {str(k): float(v) for k, v in g.items() if np.isfinite(v)}
    te_globals[te_col] = float(train_df[te_col].mean())
    print(f"  TE map {te_col:24s} {len(te_maps[te_col]):>5} categories", flush=True)

# 2b. label-encoder maps for the raw categorical features that feed the model directly
CAT_FEATURES = [f for f in FEATURES if DF[f].dtype == object]
cat_maps = {}
for c in CAT_FEATURES:
    cats = sorted(DF[c].astype(str).fillna("__NA__").unique())
    cat_maps[c] = {v: i for i, v in enumerate(cats)}
print(f"[{time.time()-t0:.0f}s] categorical features label-encoded: {CAT_FEATURES}", flush=True)

# 2c. encode the matrix exactly as the API will
X = DF[FEATURES].copy()
for c in CAT_FEATURES:
    X[c] = X[c].astype(str).fillna("__NA__").map(cat_maps[c]).astype(float)
X = X.apply(pd.to_numeric, errors="coerce")
medians = {c: (float(X[c].iloc[tr].median()) if np.isfinite(X[c].iloc[tr].median()) else 0.0)
           for c in FEATURES}
Xv = X.to_numpy(dtype=np.float32)

# ---------------------------------------------------------------- 3. train
model = xgb.XGBClassifier(
    n_estimators=600, learning_rate=0.05, max_depth=6,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
    reg_lambda=1.0, objective="binary:logistic", eval_metric="auc",
    tree_method="hist", n_jobs=N_JOBS, random_state=SEED,
    early_stopping_rounds=50,
)
model.fit(Xv[tr], y[tr], eval_set=[(Xv[va], y[va])], verbose=False)
best_it = model.best_iteration
print(f"[{time.time()-t0:.0f}s] trained; best_iteration={best_it}", flush=True)

p_test = model.predict_proba(Xv[te])[:, 1]
p_val = model.predict_proba(Xv[va])[:, 1]
auroc = roc_auc_score(y[te], p_test)
ap = average_precision_score(y[te], p_test)
brier = brier_score_loss(y[te], p_test)


def ece(yy, pp, nb=10):
    bins = np.linspace(0, 1, nb + 1); idx = np.digitize(pp, bins) - 1; e = 0.0
    for b in range(nb):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(pp[m].mean() - yy[m].mean())
    return float(e)


# Youden-J operating threshold on the validation split (never the test set)
from sklearn.metrics import roc_curve
fpr, tpr, thr = roc_curve(y[va], p_val)
threshold = float(thr[np.argmax(tpr - fpr)])

metrics = {
    "test_auroc": float(auroc), "test_average_precision": float(ap),
    "test_brier": float(brier), "test_ece": ece(y[te], p_test),
    "mean_predicted": float(p_test.mean()), "observed_rate": float(y[te].mean()),
    "operating_threshold": threshold, "best_iteration": int(best_it),
    "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
}
print(json.dumps(metrics, indent=2), flush=True)

# risk-tier cut-points from validation-set quantiles
tiers = {"low_max": float(np.quantile(p_val, 0.50)),
         "moderate_max": float(np.quantile(p_val, 0.80)),
         "high_max": float(np.quantile(p_val, 0.95))}

# ---------------------------------------------------------------- 4. persist
model.get_booster().save_model(str(ART / "model.json"))
joblib.dump({
    "feature_order": FEATURES,
    "categorical_features": CAT_FEATURES,
    "cat_maps": cat_maps,
    "te_maps": te_maps,
    "te_globals": te_globals,
    "te_pairs": TE_PAIRS,
    "medians": medians,
    "threshold": threshold,
    "tiers": tiers,
}, ART / "serving_artifacts.joblib", compress=3)
(ART / "metadata.json").write_text(json.dumps({
    "model": "XGBClassifier",
    "feature_set": "RFE-67",
    "n_features": len(FEATURES),
    "features": FEATURES,
    "metrics": metrics,
    "tiers": tiers,
    "training_data": "MIMIC-IV v3.1 Medicare cohort (244,576 admissions)",
    "split": "patient-grouped (subject_id), identical to the manuscript",
    "seed": SEED,
}, indent=2))
print(f"[{time.time()-t0:.0f}s] artifacts written to {ART}", flush=True)
