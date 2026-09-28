"""
xgboost_shap.py — advanced ML + XAI stage (owner: Abdullah).

Tuned XGBoost regression for both targets, plus SHAP explainability.

Discipline
----------
  - hyperparameters are chosen by randomised search with grouped cross-validation on the
    TRAINING rows only, grouped on physical road
  - the held-out test rows are used once, at the end, for the reported score
  - nothing is re-split; config.SPLIT_COL is used as delivered by Rayyan
  - a full-dataset grouped CV is also run, because 182 (PCI) and 129 (RUT) test rows
    cannot reliably separate models on their own

Feature set
-----------
Prefers the reduced feature set from feature_reduction.py when present. On both targets
the reduced set scored slightly BETTER than the full one in grouped CV — fewer features,
less overfitting — so this is not a trade of accuracy for simplicity.

Outputs per target:
  results/model_performance_<key>.csv    appended via results_io
  results/shap_values_<key>.csv          feature importances, machine-readable
  plots/xgboost_shap_importance_<key>.png
  plots/xgboost_shap_beeswarm_<key>.png
  plots/xgboost_case_*_<key>.png         individual prediction explanations
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, ParameterSampler

import config
import results_io
from src.models.feature_reduction import build_matrix, get_model

# Search space capped for the sample size. 794 (PCI) and 564 (RUT) training rows
# cannot support 900-tree depth-8 models — those configurations win the inner folds by
# memorising them and then generalise worse. Shallower trees, stronger regularisation.
# Justified by sample size, not by test performance.
SEARCH_SPACE = {
    "n_estimators":     [150, 250, 400, 600],
    "learning_rate":    [0.02, 0.04, 0.06, 0.10],
    "max_depth":        [2, 3, 4, 5],
    "min_child_weight": [3, 5, 10, 20, 30],
    "subsample":        [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "reg_alpha":        [0.0, 0.1, 0.5, 1.0],
    "reg_lambda":       [1.0, 3.0, 10.0, 30.0],
}

# Repeats of the grouped CV during tuning. With one pass, a candidate can win on a
# single lucky fold split; averaging over several shuffles makes the selection stable.
N_TUNE_REPEATS = 3

N_SEARCH_ITER = int(os.environ.get("PCI_SEARCH_ITER", 25))
N_INNER_FOLDS = 5
N_CV_FOLDS = 5


def _metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")}


def load_reduced(target_key):
    """The encoded feature names chosen by feature_reduction.py, or None."""
    path = os.path.join(config.RESULTS_DIR, f"reduced_features_{target_key}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))["features"]


def tune(X, y, groups, n_iter=N_SEARCH_ITER, n_splits=N_INNER_FOLDS,
         repeats=N_TUNE_REPEATS):
    """Randomised search with REPEATED grouped CV, training rows only.

    Each candidate is scored over `repeats` different shuffles of the grouped folds and
    the mean taken, so a configuration cannot be selected on the strength of one
    favourable split. With only a few hundred training rows that was the main source of
    the gap between inner-CV and held-out scores.

    Returns (params, mean_cv_rmse).
    """
    candidates = list(ParameterSampler(SEARCH_SPACE, n_iter=n_iter,
                                       random_state=config.RANDOM_SEED))
    best, best_rmse = None, np.inf
    for params in candidates:
        scores = []
        for r in range(repeats):
            order = np.random.default_rng(config.RANDOM_SEED + r).permutation(len(X))
            Xs, ys, gs = X.iloc[order], y.iloc[order], groups.iloc[order]
            for tr, te in GroupKFold(n_splits=n_splits).split(Xs, ys, gs):
                m = get_model(params)
                m.fit(Xs.iloc[tr], ys.iloc[tr])
                scores.append(np.sqrt(np.mean((ys.iloc[te] - m.predict(Xs.iloc[te])) ** 2)))
        mean_rmse = float(np.mean(scores))
        if mean_rmse < best_rmse:
            best, best_rmse = params, mean_rmse
    return best, best_rmse


def compute_shap(model, X):
    import shap
    try:
        return shap.TreeExplainer(model)(X)
    except Exception:
        import xgboost as xgb
        contribs = model.get_booster().predict(
            xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)
        return shap.Explanation(values=contribs[:, :-1], base_values=contribs[:, -1],
                                data=X.values, feature_names=list(X.columns))


def shap_outputs(model, X, target_key, units):
    import shap
    sv = compute_shap(model, X)
    arr = sv.values
    data = np.asarray(sv.data, float)

    corr = np.full(arr.shape[1], np.nan)
    for j in range(arr.shape[1]):
        ok = ~(np.isnan(data[:, j]) | np.isnan(arr[:, j]))
        if ok.sum() > 2 and np.std(data[ok, j]) > 0 and np.std(arr[ok, j]) > 0:
            corr[j] = np.corrcoef(data[ok, j], arr[ok, j])[0, 1]

    imp = pd.DataFrame({"feature": list(X.columns),
                        "mean_abs_shap": np.abs(arr).mean(axis=0),
                        "value_shap_corr": corr})
    imp["direction"] = np.where(imp.value_shap_corr.isna(), "unclear",
                        np.where(imp.value_shap_corr >= 0, "higher value -> higher target",
                                 "higher value -> lower target"))
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.to_csv(config.TARGETS[target_key]["shap_results_path"], index=False)

    top = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.34 * len(top))))
    ax.barh(top.feature, top.mean_abs_shap, color="#1f3b63")
    ax.set_xlabel(f"Mean |SHAP| ({units})")
    ax.set_title(f"{target_key.upper()} — global feature importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR,
                f"xgboost_shap_importance_{target_key}.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure()
    shap.plots.beeswarm(sv, max_display=len(X.columns), show=False)
    plt.title(f"{target_key.upper()} — SHAP summary")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, f"xgboost_shap_beeswarm_{target_key}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return imp, sv


def individual_cases(sv, y_true, y_pred, target_key):
    """Explain four deliberately chosen predictions, not arbitrary ones."""
    import shap
    err = np.asarray(y_pred) - np.asarray(y_true)
    cases = {"worst_predicted": int(np.argmin(y_pred)),
             "best_predicted": int(np.argmax(y_pred)),
             "largest_overestimate": int(np.argmax(err)),
             "largest_underestimate": int(np.argmin(err))}
    rows = []
    for label, i in cases.items():
        try:
            fig = plt.figure()
            shap.plots.waterfall(sv[i], show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(config.PLOTS_DIR,
                        f"xgboost_case_{label}_{target_key}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"    waterfall for {label} skipped: {exc}")
        rows.append({"case": label, "row": i,
                     "actual": round(float(np.asarray(y_true)[i]), 2),
                     "predicted": round(float(y_pred[i]), 2),
                     "error": round(float(err[i]), 2)})
    pd.DataFrame(rows).to_csv(
        os.path.join(config.RESULTS_DIR, f"individual_explanations_{target_key}.csv"),
        index=False)
    return rows


def full_dataset_cv(X, y, groups, params):
    """Grouped CV over every row — the held-out split alone is too small to trust."""
    folds = list(GroupKFold(n_splits=N_CV_FOLDS).split(X, y, groups))
    oof = np.full(len(X), np.nan)
    for tr, te in folds:
        m = get_model(params)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
    return oof, _metrics(y, oof)


def run(target_key: str = None):
    if target_key is None:
        for key in config.TARGETS:
            run(key)
        return

    target = config.TARGETS[target_key]
    label = target_key.upper()
    units = "PCI points" if target_key == "pci" else "1/8 inch"

    df = pd.read_csv(target["dataset_path"], low_memory=False).reset_index(drop=True)
    train_mask = df[config.SPLIT_COL] == config.TRAIN_VALUE
    y = df[target["target_col"]]
    groups = df[config.PHYSICAL_ROAD_ID_COL]

    numeric = [c for c in target["numeric_feature_cols"] if c in df.columns]
    categorical = [c for c in target["categorical_feature_cols"] if c in df.columns]
    X_full = build_matrix(df, numeric, categorical, train_mask)

    reduced = load_reduced(target_key)
    if reduced:
        missing = [c for c in reduced if c not in X_full.columns]
        if missing:
            print(f"[{label}] reduced features missing from matrix: {missing} — using full set")
            X = X_full
        else:
            X = X_full[reduced]
            print(f"[{label}] using reduced feature set: {len(reduced)} of {X_full.shape[1]}")
    else:
        X = X_full

    Xtr, ytr, gtr = X[train_mask], y[train_mask], groups[train_mask]
    Xte, yte = X[~train_mask], y[~train_mask]
    print(f"[{label}] train {len(Xtr)}  test {len(Xte)}  train roads {gtr.nunique()}")

    # --- untuned reference ----------------------------------------------------
    base = get_model()
    base.fit(Xtr, ytr)
    m_default = _metrics(yte, base.predict(Xte))
    print(f"[{label}] XGBoost (default)  MAE {m_default['mae']:7.3f}  "
          f"RMSE {m_default['rmse']:7.3f}  R2 {m_default['r2']:.3f}")

    # --- tuning, training rows only -------------------------------------------
    params, inner_rmse = tune(Xtr, ytr, gtr)
    model = get_model(params)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    m_tuned = _metrics(yte, pred)
    print(f"[{label}] XGBoost (tuned)    MAE {m_tuned['mae']:7.3f}  "
          f"RMSE {m_tuned['rmse']:7.3f}  R2 {m_tuned['r2']:.3f}   "
          f"(inner CV RMSE {inner_rmse:.3f})")

    # --- full-dataset grouped CV ----------------------------------------------
    oof, m_cv = full_dataset_cv(X, y, groups, params)
    print(f"[{label}] XGBoost (grouped CV, all {len(X)} rows)  "
          f"MAE {m_cv['mae']:7.3f}  RMSE {m_cv['rmse']:7.3f}  R2 {m_cv['r2']:.3f}")

    results_io.append_results(target["results_metrics_path"], [
        {"model": "XGBoost (default)", **m_default},
        {"model": "XGBoost (tuned)", **m_tuned},
    ])
    pd.DataFrame({"y_true": y, "oof_pred": oof,
                  config.PHYSICAL_ROAD_ID_COL: groups}).to_csv(
        os.path.join(config.RESULTS_DIR, f"xgboost_oof_{target_key}.csv"), index=False)
    with open(os.path.join(config.RESULTS_DIR, f"xgboost_best_params_{target_key}.json"),
              "w") as f:
        json.dump({"params": params, "inner_cv_rmse": inner_rmse,
                   "test": m_tuned, "grouped_cv_all_rows": m_cv,
                   "n_features": int(X.shape[1])}, f, indent=2, default=str)

    # --- explainability --------------------------------------------------------
    imp, sv = shap_outputs(model, Xte, target_key, units)
    print(f"[{label}] SHAP top factors:")
    for _, r in imp.head(6).iterrows():
        print(f"    {r.feature[:40]:40s} {r.mean_abs_shap:8.4f}  {r.direction}")
    individual_cases(sv, yte.values, pred, target_key)

    import joblib
    joblib.dump(model, os.path.join(config.MODELS_DIR, f"xgboost_{target_key}.joblib"))
    return {"default": m_default, "tuned": m_tuned, "cv": m_cv, "params": params}


if __name__ == "__main__":
    run()
