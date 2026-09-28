"""
compare_vs_baselines.py — head-to-head against Luke's baselines (owner: Abdullah).

Luke reports that no ML model beats his anchored age-trend baseline. This reimplements
that baseline so both approaches are scored on identical rows, under two protocols:

  1. the held-out test split (what Luke reported)
  2. grouped cross-validation over every row — the test split is 182 rows for PCI and
     129 for RUT, too few to separate models reliably

A paired bootstrap over test roads says whether any difference is real.

The anchored age-trend baseline: fit condition-vs-age on training rows, then predict
    next = current + (curve(age_next) - curve(age_now))
so the curve supplies the expected *change* and current condition supplies the level.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import config
from src.models.feature_reduction import build_matrix, get_model
from src.models.xgboost_shap import load_reduced, _metrics

N_FOLDS = 5
N_BOOT = 4000
CURRENT = {"pci": "PCI_SCORE", "rut": "RUT_DEPTH_1_8IN"}


def fit_age_curve(age, value):
    """Power-law-ish deterioration curve fitted in log space, as HDM-4/Austroads use."""
    ok = (~np.isnan(age)) & (~np.isnan(value)) & (age > 0)
    a, v = np.asarray(age)[ok], np.asarray(value)[ok]
    if len(a) < 10:
        return lambda x: np.full_like(np.asarray(x, float), np.nanmean(v))
    coef = np.polyfit(np.log(a), v, 1)
    return lambda x: np.polyval(coef, np.log(np.clip(np.asarray(x, float), 0.01, None)))


def anchored_trend(df_tr, df_te, key):
    cur = CURRENT[key]
    curve = fit_age_curve(df_tr["PAVEMENT_AGE_YEARS"], df_tr[cur])
    age_now = df_te["PAVEMENT_AGE_YEARS"].values
    age_next = age_now + df_te["YEARS_TO_NEXT_SURVEY"].fillna(
        df_tr["YEARS_TO_NEXT_SURVEY"].median()).values
    delta = curve(age_next) - curve(age_now)
    return df_te[cur].values + delta


def run_for_target(key):
    target = config.TARGETS[key]
    df = pd.read_csv(target["dataset_path"], low_memory=False).reset_index(drop=True)
    y = df[target["target_col"]]
    groups = df[config.PHYSICAL_ROAD_ID_COL]
    train_mask = df[config.SPLIT_COL] == config.TRAIN_VALUE

    numeric = [c for c in target["numeric_feature_cols"] if c in df.columns]
    categorical = [c for c in target["categorical_feature_cols"] if c in df.columns]
    X_full = build_matrix(df, numeric, categorical, train_mask)
    reduced = load_reduced(key)
    X = X_full[reduced] if reduced else X_full

    params = json.load(open(os.path.join(
        config.RESULTS_DIR, f"xgboost_best_params_{key}.json")))["params"]
    params = {k: (int(v) if isinstance(v, str) and v.isdigit() else
                  float(v) if isinstance(v, str) else v) for k, v in params.items()}

    lo, hi = (0, 100) if key == "pci" else (0, None)

    def clip(p):
        return np.clip(p, lo, hi) if hi else np.clip(p, lo, None)

    # ---------- protocol 1: held-out test split ----------
    tr, te = df[train_mask], df[~train_mask]
    yte = y[~train_mask].values
    preds_test = {
        "Persistence": clip(te[CURRENT[key]].values),
        "Age trend (anchored)": clip(anchored_trend(tr, te, key)),
    }
    m = get_model(params); m.fit(X[train_mask], y[train_mask])
    preds_test["XGBoost (tuned)"] = clip(m.predict(X[~train_mask]))

    # ---------- protocol 2: grouped CV over every row ----------
    folds = list(GroupKFold(n_splits=N_FOLDS).split(df, y, groups))
    oof = {k: np.full(len(df), np.nan) for k in preds_test}
    for f_tr, f_te in folds:
        d_tr, d_te = df.iloc[f_tr], df.iloc[f_te]
        oof["Persistence"][f_te] = clip(d_te[CURRENT[key]].values)
        oof["Age trend (anchored)"][f_te] = clip(anchored_trend(d_tr, d_te, key))
        mm = get_model(params); mm.fit(X.iloc[f_tr], y.iloc[f_tr])
        oof["XGBoost (tuned)"][f_te] = clip(mm.predict(X.iloc[f_te]))

    # ---------- report ----------
    rows = []
    for name in preds_test:
        t = _metrics(yte, preds_test[name])
        c = _metrics(y.values, oof[name])
        rows.append({"model": name,
                     "test_r2": round(t["r2"], 3), "test_rmse": round(t["rmse"], 3),
                     "test_mae": round(t["mae"], 3),
                     "cv_r2": round(c["r2"], 3), "cv_rmse": round(c["rmse"], 3),
                     "cv_mae": round(c["mae"], 3)})

    # paired bootstrap: XGBoost vs the anchored trend, on the CV predictions
    rng = np.random.default_rng(config.RANDOM_SEED)
    n = len(y); idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    a, b = oof["Age trend (anchored)"], oof["XGBoost (tuned)"]
    diffs = np.array([np.sqrt(np.mean((y.values[i] - a[i]) ** 2))
                      - np.sqrt(np.mean((y.values[i] - b[i]) ** 2)) for i in idx])
    lo_ci, hi_ci = np.percentile(diffs, [2.5, 97.5])
    verdict = ("XGBoost better" if lo_ci > 0 else
               "baseline better" if hi_ci < 0 else "not distinguishable")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(config.RESULTS_DIR, f"comparison_vs_baselines_{key}.csv"),
               index=False)
    print(f"\n=== {key.upper()} ===")
    print(out.to_string(index=False))
    print(f"XGBoost vs anchored trend (grouped CV, RMSE gain): "
          f"{diffs.mean():+.3f}  95% CI [{lo_ci:+.3f}, {hi_ci:+.3f}]  -> {verdict}")
    return {"table": rows, "bootstrap": {"mean": float(diffs.mean()),
            "ci_low": float(lo_ci), "ci_high": float(hi_ci), "verdict": verdict}}


def run():
    res = {k: run_for_target(k) for k in config.TARGETS}
    with open(os.path.join(config.RESULTS_DIR, "comparison_vs_baselines.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    run()
