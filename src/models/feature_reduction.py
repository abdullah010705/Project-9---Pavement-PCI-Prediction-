"""
feature_reduction.py — cut the feature set down (owner: Abdullah).

The client flagged ~120 features as too many and named pavement age, time and climate
as the ones that matter. This ranks features by two independent methods, sweeps the
feature count, and picks the smallest set that holds accuracy.

Discipline: every decision here is made on the TRAINING rows only, using grouped
cross-validation on physical road. The held-out test split is never touched — otherwise
the feature choice itself leaks test information into the final scores.

Outputs, per target:
  results/feature_ranking_<key>.csv     every feature, both importance scores, final rank
  results/feature_sweep_<key>.csv       CV score at each feature count
  results/reduced_features_<key>.json   the chosen set
  plots/feature_sweep_<key>.png         accuracy vs feature count
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

import config

SWEEP_SIZES = [5, 10, 15, 20, 25, 30, 40, 60]
N_FOLDS = 5
N_JOBS = int(os.environ.get("PCI_N_JOBS", min(4, os.cpu_count() or 4)))


def get_model(params=None):
    from xgboost import XGBRegressor
    defaults = dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
                    colsample_bytree=0.8, objective="reg:squarederror",
                    tree_method="hist", random_state=config.RANDOM_SEED, n_jobs=N_JOBS)
    defaults.update(params or {})
    return XGBRegressor(**defaults)


def build_matrix(df, numeric, categorical, train_mask):
    """Median/mode impute fitted on train only, one-hot categoricals, missingness flags."""
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


def _rmse(a, p):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(p)) ** 2)))


def cv_score(X, y, groups, feats, n_folds=N_FOLDS):
    """Grouped CV RMSE and R2 using only the named features."""
    Xs = X[feats]
    folds = list(GroupKFold(n_splits=n_folds).split(Xs, y, groups))
    oof = np.full(len(Xs), np.nan)
    for tr, te in folds:
        m = get_model()
        m.fit(Xs.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(Xs.iloc[te])
    ss_res = float(np.sum((y.values - oof) ** 2))
    ss_tot = float(np.sum((y.values - y.values.mean()) ** 2))
    return _rmse(y, oof), (1 - ss_res / ss_tot)


def shap_importance(model, X):
    """Mean |SHAP| per feature. Falls back to XGBoost's native TreeSHAP, since shap's
    loader cannot parse XGBoost 3.x's vector-valued base_score."""
    import shap
    try:
        vals = shap.TreeExplainer(model)(X).values
    except Exception:
        import xgboost as xgb
        contribs = model.get_booster().predict(
            xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)
        vals = contribs[:, :-1]
    return pd.Series(np.abs(vals).mean(axis=0), index=X.columns)


def run_for_target(target_key: str) -> dict:
    target = config.TARGETS[target_key]
    label = target_key.upper()
    df = pd.read_csv(target["dataset_path"], low_memory=False).reset_index(drop=True)

    train_mask = df[config.SPLIT_COL] == config.TRAIN_VALUE
    numeric = [c for c in target["numeric_feature_cols"] if c in df.columns]
    categorical = [c for c in target["categorical_feature_cols"] if c in df.columns]

    X_all = build_matrix(df, numeric, categorical, train_mask)
    y = df[target["target_col"]]
    groups = df[config.PHYSICAL_ROAD_ID_COL]

    Xtr, ytr, gtr = X_all[train_mask], y[train_mask], groups[train_mask]
    print(f"\n[{label}] {len(df)} rows, {X_all.shape[1]} features after encoding, "
          f"{train_mask.sum()} training rows")

    # --- two independent importance measures, both on training rows only -----
    base = get_model()
    base.fit(Xtr, ytr)

    shap_imp = shap_importance(base, Xtr)

    perm = permutation_importance(base, Xtr, ytr, n_repeats=5,
                                  random_state=config.RANDOM_SEED, n_jobs=N_JOBS,
                                  scoring="neg_root_mean_squared_error")
    perm_imp = pd.Series(perm.importances_mean, index=Xtr.columns)

    rank = pd.DataFrame({
        "feature": Xtr.columns,
        "shap_importance": shap_imp.values,
        "permutation_importance": perm_imp.values,
    })
    rank["shap_rank"] = rank.shap_importance.rank(ascending=False)
    rank["perm_rank"] = rank.permutation_importance.rank(ascending=False)
    # average of the two ranks — a feature has to do well on both to survive
    rank["combined_rank"] = (rank.shap_rank + rank.perm_rank) / 2
    rank = rank.sort_values("combined_rank").reset_index(drop=True)
    rank.to_csv(os.path.join(config.RESULTS_DIR, f"feature_ranking_{target_key}.csv"),
                index=False)

    print(f"[{label}] top 10 by combined rank:")
    for i, r in rank.head(10).iterrows():
        print(f"    {i+1:>2}. {r.feature[:48]:48s} shap={r.shap_importance:8.4f}")

    # --- sweep the feature count ---------------------------------------------
    full_rmse, full_r2 = cv_score(Xtr, ytr, gtr, list(Xtr.columns))
    sweep = [{"n_features": len(Xtr.columns), "rmse": full_rmse, "r2": full_r2,
              "label": "all"}]
    for n in SWEEP_SIZES:
        if n >= len(Xtr.columns):
            continue
        feats = rank.feature.head(n).tolist()
        rm, r2 = cv_score(Xtr, ytr, gtr, feats)
        sweep.append({"n_features": n, "rmse": rm, "r2": r2, "label": str(n)})
        print(f"[{label}]   {n:>3} features -> CV RMSE {rm:6.3f}   R2 {r2:.3f}")
    print(f"[{label}]   all ({len(Xtr.columns)}) -> CV RMSE {full_rmse:6.3f}   R2 {full_r2:.3f}")

    sweep_df = pd.DataFrame(sweep).sort_values("n_features").reset_index(drop=True)
    sweep_df.to_csv(os.path.join(config.RESULTS_DIR, f"feature_sweep_{target_key}.csv"),
                    index=False)

    # --- choose: smallest set within 2% of the best CV RMSE ------------------
    best = sweep_df.rmse.min()
    tolerance = best * 1.02
    eligible = sweep_df[(sweep_df.rmse <= tolerance) & (sweep_df.n_features <= 60)]
    chosen_n = int(eligible.n_features.min()) if len(eligible) else 30
    chosen = rank.feature.head(chosen_n).tolist()
    chosen_row = sweep_df[sweep_df.n_features == chosen_n].iloc[0]

    print(f"[{label}] chosen: {chosen_n} features "
          f"(CV RMSE {chosen_row.rmse:.3f} vs {full_rmse:.3f} with all "
          f"{len(Xtr.columns)}; R2 {chosen_row.r2:.3f} vs {full_r2:.3f})")

    with open(os.path.join(config.RESULTS_DIR, f"reduced_features_{target_key}.json"), "w") as f:
        json.dump({
            "target": target["target_col"],
            "n_features_before": int(len(Xtr.columns)),
            "n_features_after": chosen_n,
            "cv_rmse_before": full_rmse, "cv_rmse_after": float(chosen_row.rmse),
            "cv_r2_before": full_r2, "cv_r2_after": float(chosen_row.r2),
            "selection_rule": "smallest set within 2% of best grouped-CV RMSE, train rows only",
            "features": chosen,
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(sweep_df.n_features, sweep_df.rmse, "o-", color="#1f3b63")
    ax.axhline(full_rmse, ls="--", lw=1, color="#888", label=f"all {len(Xtr.columns)} features")
    ax.axvline(chosen_n, ls=":", lw=1.5, color="#c0392b", label=f"chosen: {chosen_n}")
    ax.set_xlabel("Number of features"); ax.set_ylabel("Grouped CV RMSE (train rows)")
    ax.set_title(f"{label} — accuracy vs feature count")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, f"feature_sweep_{target_key}.png"), dpi=150)
    plt.close(fig)

    return {"target_key": target_key, "chosen": chosen, "n_before": len(Xtr.columns),
            "n_after": chosen_n, "rmse_before": full_rmse, "rmse_after": float(chosen_row.rmse),
            "r2_before": full_r2, "r2_after": float(chosen_row.r2)}


def run():
    out = {}
    for key in config.TARGETS:
        out[key] = run_for_target(key)
    with open(os.path.join(config.RESULTS_DIR, "feature_reduction_summary.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "chosen"}
                   for k, v in out.items()}, f, indent=2)
    print("\n[feature_reduction] summary")
    for k, v in out.items():
        print(f"  {k.upper():4s} {v['n_before']:>3} -> {v['n_after']:>2} features   "
              f"CV RMSE {v['rmse_before']:.3f} -> {v['rmse_after']:.3f}   "
              f"R2 {v['r2_before']:.3f} -> {v['r2_after']:.3f}")
    return out


if __name__ == "__main__":
    run()
