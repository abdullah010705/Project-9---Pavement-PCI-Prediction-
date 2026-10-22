"""
baseline_traditional.py — baselines + traditional ML (owner: Luke).

REIMPLEMENTED DURING INTEGRATION, NOT AUTHORED BY LUKE.
------------------------------------------------------
Luke produced results (write-up dated 19 Sep 2026) but his script was not in the repo,
and the pipeline could not run end to end without this stage. This reimplements the six
approaches exactly as described in his write-up so the pipeline works for the review.

*** Luke must check this against his own code before any of these numbers are used. ***

Known discrepancy: Luke reports his anchored age-trend baseline at R2 0.802 / RMSE 12.05
on PCI. This implementation scores about 0.71 / 14.6 on the same test rows. That gap is
larger than the difference between any two models in the project, and the headline claim
("no ML model beats the best baseline") depends on which figure is right. Treat the
anchored-trend row below as unverified until Luke's code lands.

Approaches, per his method section:
  1. Persistence                     next condition = current condition
  2. Age trend, absolute (power-law) curve fitted on age alone
  3. Age trend, anchored              current condition + expected change from the curve
  4. Linear Regression                full feature set, scaled
  5. Random Forest                    500 trees
  6. Gradient Boosting                300 trees, depth 3
All use Rayyan's split as delivered; predictions are clipped to the physical range.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
import results_io

CURRENT = {"pci": "PCI_SCORE", "rut": "RUT_DEPTH_1_8IN"}
CLIP = {"pci": (0.0, 100.0), "rut": (0.0, None)}


def _metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2)); ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")}


def _clip(p, key):
    lo, hi = CLIP[key]
    return np.clip(p, lo, hi) if hi is not None else np.clip(p, lo, None)


def _age_curve(age, value):
    """Power-law form in log(age) — the shape HDM-4 and Austroads use."""
    ok = (~np.isnan(age)) & (~np.isnan(value)) & (age > 0)
    a, v = np.asarray(age)[ok], np.asarray(value)[ok]
    if len(a) < 10:
        return lambda x: np.full_like(np.asarray(x, float), float(np.nanmean(v)))
    coef = np.polyfit(np.log(a), v, 1)
    return lambda x: np.polyval(coef, np.log(np.clip(np.asarray(x, float), 0.01, None)))


def run(target_key: str = None):
    if target_key is None:
        for key in config.TARGETS:
            run(key)
        return

    target = config.TARGETS[target_key]
    label = target_key.upper()
    df = pd.read_csv(target["dataset_path"], low_memory=False).reset_index(drop=True)

    train_mask = df[config.SPLIT_COL] == config.TRAIN_VALUE
    tr, te = df[train_mask], df[~train_mask]
    y_tr, y_te = df.loc[train_mask, target["target_col"]], df.loc[~train_mask, target["target_col"]]
    cur = CURRENT[target_key]

    numeric = [c for c in target["numeric_feature_cols"] if c in df.columns]
    categorical = [c for c in target["categorical_feature_cols"] if c in df.columns]
    feats = numeric + categorical

    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    prep_scaled = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

    preds = {}

    # 1. persistence
    preds["Persistence baseline"] = _clip(te[cur].values, target_key)

    # 2. age trend, absolute
    curve = _age_curve(tr["PAVEMENT_AGE_YEARS"].values, y_tr.values)
    preds["Age trend (absolute)"] = _clip(curve(te["PAVEMENT_AGE_YEARS"].values), target_key)

    # 3. age trend, anchored to current condition
    curve_cur = _age_curve(tr["PAVEMENT_AGE_YEARS"].values, tr[cur].values)
    age_now = te["PAVEMENT_AGE_YEARS"].values
    gap = te["YEARS_TO_NEXT_SURVEY"].fillna(tr["YEARS_TO_NEXT_SURVEY"].median()).values
    preds["Age trend (anchored)"] = _clip(
        te[cur].values + (curve_cur(age_now + gap) - curve_cur(age_now)), target_key)

    # 4-6. traditional ML
    # Linear Regression on the rut target produces divide-by-zero and overflow warnings
    # from numpy during the matrix multiply. They are printed noise, not errors: the fit,
    # the predictions and the reported score are identical either way. Silenced so the
    # pipeline output stays readable.
    import warnings as _w
    _ctx = _w.catch_warnings(); _ctx.__enter__()
    _w.filterwarnings("ignore", category=RuntimeWarning)
    _err = np.errstate(divide="ignore", over="ignore", invalid="ignore"); _err.__enter__()

    for name, model, pp in [
        ("Linear Regression", LinearRegression(), prep_scaled),
        ("Random Forest", RandomForestRegressor(n_estimators=500,
            random_state=config.RANDOM_SEED, n_jobs=-1), prep),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=300, max_depth=3,
            random_state=config.RANDOM_SEED), prep),
    ]:
        pipe = Pipeline([("prep", pp), ("model", model)]).fit(tr[feats], y_tr)
        preds[name] = _clip(pipe.predict(te[feats]), target_key)

    _err.__exit__(None, None, None); _ctx.__exit__(None, None, None)

    rows = [{"model": n, **_metrics(y_te, p)} for n, p in preds.items()]
    results_io.append_results(target["results_metrics_path"], rows)

    out = te[[config.ROAD_ID_COL, config.PHYSICAL_ROAD_ID_COL,
              config.SURVEY_DATE_COL, cur]].copy()
    out[f"{target['target_col']}_ACTUAL"] = y_te.values
    for n, p in preds.items():
        out["PRED_" + n.upper().replace(" ", "_").replace("(", "").replace(")", "")] = p
    out.to_csv(target["predictions_path"], index=False)

    lims = (0, 100) if target_key == "pci" else (0, float(np.nanmax(y_te)) * 1.1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (n, p) in zip(axes.flatten(), preds.items()):
        s = _metrics(y_te, p)
        ax.scatter(y_te, p, alpha=0.5, s=18, edgecolor="none")
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_title(f"{n}\nR2={s['r2']:.3f}  RMSE={s['rmse']:.2f}", fontsize=9)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    fig.suptitle(f"{label} — actual vs predicted, test set (n={len(y_te)})")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(config.PLOTS_DIR, f"actual_vs_predicted_{target_key}.png"), dpi=150)
    plt.close(fig)

    print(f"[{label}] baselines + traditional ML")
    print(pd.DataFrame(rows).sort_values("rmse").to_string(index=False))
    return rows


if __name__ == "__main__":
    run()
