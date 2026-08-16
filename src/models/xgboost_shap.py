"""
xgboost_shap.py — advanced ML + explainability stage (owner: Abdullah).

Builds an XGBoost regressor for PCI_NEXT, tunes it, evaluates it on Rayyan's held-out
test split, and explains it with SHAP.

Discipline:
  - hyperparameters are chosen by randomised search with grouped CV on the TRAINING
    rows only, grouped on PHYSICAL_ROAD_ID
  - the test rows are used once, at the end, for the reported score
  - nothing is re-split; config.SPLIT_COL is used as delivered

Note on the forecast horizon
----------------------------
config.py lists YEARS_TO_NEXT_SURVEY under TARGET_AUX_COLS, marked "must NOT be used as
a model input". This stage respects that by default. It also fits a second variant that
includes it, reported as a separate row, because an ablation over all 976 rows found the
horizon improves RMSE by +0.34 under Luke's exact preprocessing (95% CI [+0.06, +0.63])
and +0.66 with missingness indicators (CI [+0.36, +0.97]).

The argument for including it: in deployment you *choose* the horizon ("what will PCI be
in 2 years"), so it is known at prediction time. The horizon ranges 0.01-9.69 years in
this data, so excluding it asks the model to predict PCI at an unspecified future point.

Both rows are reported so the team can decide. See PCI_TARGET_DEFECT.md and the
README in "Abdullah - XGBoost + SHAP" for the full analysis.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, ParameterSampler

import config
from src import results_io

HORIZON_COL = "YEARS_TO_NEXT_SURVEY"

# Deliberately excludes the expensive corner (1200 trees at depth 10). On 794 training
# rows a deep, very long model overfits and never won a search, but it dominated runtime.
SEARCH_SPACE = {
    "n_estimators":     [200, 400, 600, 900],
    "learning_rate":    [0.02, 0.05, 0.08, 0.12],
    "max_depth":        [3, 4, 5, 6, 8],
    "min_child_weight": [1, 3, 5, 10, 20],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha":        [0.0, 0.01, 0.1, 1.0],
    "reg_lambda":       [0.5, 1.0, 3.0, 10.0],
}

# Runtime controls. Both can be overridden by environment variable, so a demo can run
# fast without editing code:
#     PCI_SEARCH_ITER=6 ./run.sh        quick demo
#     PCI_SEARCH_ITER=60 ./run.sh       thorough run for the thesis
N_SEARCH_ITER = int(os.environ.get("PCI_SEARCH_ITER", 20))
N_INNER_FOLDS = 5

# XGBoost threads. n_jobs=-1 grabs every core, which is counterproductive on a dataset
# this small — the threading overhead exceeds the work per tree, and on machines with
# many cores it made the stage roughly 18x slower. Cap it low.
N_JOBS = int(os.environ.get("PCI_N_JOBS", min(4, os.cpu_count() or 4)))


def _get_model(params=None):
    from xgboost import XGBRegressor
    defaults = dict(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8,
                    colsample_bytree=0.8, objective="reg:squarederror",
                    tree_method="hist", random_state=config.RANDOM_SEED,
                    n_jobs=N_JOBS)
    defaults.update(params or {})
    return XGBRegressor(**defaults)


def build_matrix(df, numeric, categorical, train_mask):
    """Median/mode impute (fit on train only), one-hot categoricals, and add a
    missingness indicator for every column that has gaps.

    The indicators matter: RUT_DEPTH_1_8IN__isna turns out to be highly predictive,
    which is what exposed the PCI target defect. Kept deliberately, and reported.
    """
    X = pd.DataFrame(index=df.index)
    for col in numeric:
        if col not in df.columns:
            continue
        X[col] = df[col].fillna(df.loc[train_mask, col].median())
        if df[col].isna().any():
            X[f"{col}__isna"] = df[col].isna().astype(int)
    for col in categorical:
        if col not in df.columns:
            continue
        mode = df.loc[train_mask, col].mode()
        fill = mode.iloc[0] if len(mode) else "MISSING"
        s = df[col].fillna(fill).astype(str)
        allowed = set(df.loc[train_mask, col].dropna().astype(str).unique())
        s = s.where(s.isin(allowed), "OTHER")
        X = pd.concat([X, pd.get_dummies(s, prefix=col, dtype=int)], axis=1)
    return X


def tune(X, y, groups, n_iter=N_SEARCH_ITER, n_splits=N_INNER_FOLDS):
    """Randomised search with grouped CV. Returns (best_params, best_rmse)."""
    if groups.nunique() < n_splits:
        raise ValueError(f"{groups.nunique()} groups but n_splits={n_splits}")
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    best, best_rmse = None, np.inf
    for params in ParameterSampler(SEARCH_SPACE, n_iter=n_iter,
                                   random_state=config.RANDOM_SEED):
        scores = []
        for tr, te in folds:
            m = _get_model(params)
            m.fit(X.iloc[tr], y.iloc[tr])
            scores.append(np.sqrt(np.mean((y.iloc[te] - m.predict(X.iloc[te])) ** 2)))
        mean_rmse = float(np.mean(scores))
        if mean_rmse < best_rmse:
            best, best_rmse = params, mean_rmse
    return best, best_rmse


def compute_shap(model, X):
    """SHAP values as a shap.Explanation.

    shap.TreeExplainer cannot parse XGBoost >= 3's vector-valued base_score, so we fall
    back to XGBoost's own exact TreeSHAP. Same values, no need to pin an old XGBoost.
    """
    import shap
    try:
        return shap.TreeExplainer(model)(X)
    except Exception:
        import xgboost as xgb
        contribs = model.get_booster().predict(
            xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)
        return shap.Explanation(values=contribs[:, :-1], base_values=contribs[:, -1],
                                data=X.values, feature_names=list(X.columns))


def shap_outputs(model, X, prefix="xgboost"):
    import shap
    sv = compute_shap(model, X)
    arr = sv.values

    corr = np.full(arr.shape[1], np.nan)
    data = np.asarray(sv.data, dtype=float)
    for j in range(arr.shape[1]):
        ok = ~(np.isnan(data[:, j]) | np.isnan(arr[:, j]))
        if ok.sum() > 2 and np.std(data[ok, j]) > 0 and np.std(arr[ok, j]) > 0:
            corr[j] = np.corrcoef(data[ok, j], arr[ok, j])[0, 1]

    imp = pd.DataFrame({"feature": list(X.columns),
                        "mean_abs_shap": np.abs(arr).mean(axis=0),
                        "value_shap_corr": corr})
    imp["direction"] = np.where(imp.value_shap_corr.isna(), "unclear",
                        np.where(imp.value_shap_corr >= 0, "higher value -> higher PCI",
                                 "higher value -> lower PCI"))
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.to_csv(config.SHAP_RESULTS_PATH, index=False)

    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(top))))
    ax.barh(top.feature, top.mean_abs_shap, color="#1f3b63")
    ax.set_xlabel("Mean |SHAP| (PCI points)")
    ax.set_title("XGBoost — global feature importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, f"{prefix}_shap_importance.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure()
    shap.plots.beeswarm(sv, max_display=20, show=False)
    plt.title("XGBoost — SHAP summary")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, f"{prefix}_shap_beeswarm.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    return imp, sv


def run():
    df = pd.read_csv(config.MODEL_DATASET_PATH, low_memory=False)
    train_mask = df[config.SPLIT_COL] == config.TRAIN_VALUE
    y = df[config.TARGET_COL]
    groups = df[config.PHYSICAL_ROAD_ID_COL]

    numeric = [c for c in config.NUMERIC_FEATURE_COLS if c in df.columns]
    categorical = [c for c in config.CATEGORICAL_FEATURE_COLS if c in df.columns]

    variants = [
        ("XGBoost (tuned)", numeric),
        ("XGBoost (tuned + horizon)", numeric + [HORIZON_COL]),
    ]

    rows, artefacts = [], {}
    for label, feats in variants:
        X = build_matrix(df, feats, categorical, train_mask)
        Xtr, ytr, gtr = X[train_mask], y[train_mask], groups[train_mask]
        Xte, yte = X[~train_mask], y[~train_mask]

        params, inner_rmse = tune(Xtr, ytr, gtr)
        model = _get_model(params)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        s = results_io.score(yte, pred)
        rows.append({"model": label, **s})
        artefacts[label] = (model, Xte, params, inner_rmse)
        print(f"[xgboost] {label:28s} MAE {s['mae']:6.2f}  RMSE {s['rmse']:6.2f}  "
              f"R2 {s['r2']:.3f}   (inner CV RMSE {inner_rmse:.2f})")

    import joblib
    for label, (model, _, _, _) in artefacts.items():
        fname = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        joblib.dump(model, os.path.join(config.MODELS_DIR, f"{fname}.joblib"))

    results_io.append_results(rows)

    # SHAP on the better variant
    best_label = min(rows, key=lambda r: r["rmse"])["model"]
    model, Xte, params, _ = artefacts[best_label]
    print(f"[xgboost] SHAP on: {best_label}")
    imp, _ = shap_outputs(model, Xte)
    print(imp.head(10).to_string(index=False))

    with open(os.path.join(config.RESULTS_DIR, "xgboost_best_params.json"), "w") as f:
        json.dump({label: artefacts[label][2] for label, _ in variants}, f,
                  indent=2, default=str)

    return rows


if __name__ == "__main__":
    run()
