"""
cross_validate.py — grouped cross-validation over the whole dataset.

Why this stage exists
---------------------
Every model is currently scored on Rayyan's 182-row held-out test set. That is correct
practice, but 182 rows cannot resolve differences of about 1 RMSE point — bootstrap
intervals for XGBoost vs Gradient Boosting cross zero, and the ordering of the two flips
depending on the run. Any claim that one model beats another rests on noise.

This stage re-scores every approach with 5-fold GroupKFold over all 976 rows, grouped on
PHYSICAL_ROAD_ID so no road appears in both sides of any fold. Every row is predicted
exactly once while out of training, so the comparison rests on 976 observations instead
of 182.

It also runs a paired bootstrap against the best baseline, so the results table says
whether a difference is real rather than leaving the reader to guess from three decimal
places.

Hyperparameters are held fixed at the values chosen on the training split. Re-tuning
inside every fold would be more rigorous still, but it changes what is being compared
and multiplies the runtime; noted as future work.

Outputs: results/cross_validation.csv, results/oof_predictions.csv,
         plots/cross_validation_comparison.png
"""
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from src import results_io
from src.models import baseline_traditional as bt
from src.models import xgboost_shap as xs

N_SPLITS = 5
N_BOOTSTRAP = 4000


def _sk_pipeline(kind, numeric, categorical):
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    if kind == "Linear Regression":
        prep = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ])
        model = LinearRegression()
    elif kind == "Random Forest":
        model = RandomForestRegressor(n_estimators=300, random_state=config.RANDOM_SEED,
                                      n_jobs=-1)
    elif kind == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=config.RANDOM_SEED)
    else:
        raise ValueError(kind)
    return Pipeline([("prep", prep), ("model", model)])


def run():
    df = pd.read_csv(config.MODEL_DATASET_PATH, low_memory=False).reset_index(drop=True)
    y = df[config.TARGET_COL].values
    groups = df[config.PHYSICAL_ROAD_ID_COL]

    numeric = [c for c in bt.NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in bt.CATEGORICAL_FEATURES if c in df.columns]
    xgb_numeric = [c for c in config.NUMERIC_FEATURE_COLS if c in df.columns]
    xgb_categorical = [c for c in config.CATEGORICAL_FEATURE_COLS if c in df.columns]

    best_params = {}
    params_path = os.path.join(config.RESULTS_DIR, "xgboost_best_params.json")
    if os.path.exists(params_path):
        import json
        best_params = json.load(open(params_path))

    folds = list(GroupKFold(n_splits=N_SPLITS).split(df, y, groups))
    print(f"[cross_validate] {len(df)} rows, {groups.nunique()} physical roads, "
          f"{N_SPLITS} folds, test sizes {[len(te) for _, te in folds]}")

    oof = {"Persistence baseline": df["PCI_SCORE"].values.astype(float)}

    for kind in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
        preds = np.full(len(df), np.nan)
        for tr, te in folds:
            pipe = _sk_pipeline(kind, numeric, categorical)
            pipe.fit(df.iloc[tr][numeric + categorical], y[tr])
            preds[te] = pipe.predict(df.iloc[te][numeric + categorical])
        oof[kind] = preds

    for label, extra in [("XGBoost (tuned)", []),
                         ("XGBoost (tuned + horizon)", [xs.HORIZON_COL])]:
        preds = np.full(len(df), np.nan)
        feats = xgb_numeric + extra
        for tr, te in folds:
            tmask = pd.Series(False, index=df.index)
            tmask.iloc[tr] = True
            X = xs.build_matrix(df, feats, xgb_categorical, tmask)
            m = xs._get_model(best_params.get(label))
            m.fit(X.iloc[tr], y[tr])
            preds[te] = m.predict(X.iloc[te])
        oof[label] = preds

    # ---- score + paired bootstrap against the persistence baseline ----------
    rng = np.random.default_rng(config.RANDOM_SEED)
    n = len(y)
    bidx = [rng.integers(0, n, n) for _ in range(N_BOOTSTRAP)]
    base = oof["Persistence baseline"]

    def rmse(a, p, i):
        return float(np.sqrt(np.mean((a[i] - p[i]) ** 2)))

    rows = []
    for name, p in oof.items():
        s = results_io.score(y, p)
        rec = {"model": name, "n": n, **{k: round(v, 3) for k, v in s.items()}}
        if name != "Persistence baseline":
            d = np.array([rmse(y, base, i) - rmse(y, p, i) for i in bidx])
            lo, hi = np.percentile(d, [2.5, 97.5])
            rec["rmse_gain_vs_persistence"] = round(float(d.mean()), 3)
            rec["ci_low"], rec["ci_high"] = round(float(lo), 3), round(float(hi), 3)
            rec["beats_persistence"] = "yes" if lo > 0 else (
                "no" if hi < 0 else "not distinguishable")
        rows.append(rec)

    table = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    table.to_csv(os.path.join(config.RESULTS_DIR, "cross_validation.csv"), index=False)
    pd.DataFrame({"y_true": y, **oof}).to_csv(
        os.path.join(config.RESULTS_DIR, "oof_predictions.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_df = table.sort_values("rmse", ascending=False)
    colours = ["#c0392b" if "baseline" in m.lower() else "#1f3b63"
               for m in plot_df["model"]]
    ax.barh(plot_df["model"], plot_df["rmse"], color=colours)
    ax.set_xlabel("RMSE (PCI points, lower is better)")
    ax.set_title(f"Grouped {N_SPLITS}-fold cross-validation, all {n} rows")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "cross_validation_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"[cross_validate] Grouped CV over all {n} rows (vs held-out 182):")
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    run()
