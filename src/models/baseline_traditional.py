"""
baseline_traditional.py — baselines + traditional ML stage (owner: Luke).

ASSEMBLED, NOT AUTHORED: this is Luke's pci_prediction_pipeline.py, wrapped in a run()
function and rewired to read config.py (paths, feature lists, split column) instead of
hardcoded values, so run_pipeline.py can call it and src/evaluate.py can aggregate its
output. The five approaches, the preprocessing and the seed are unchanged — Luke should
review and take ownership.

Builds, all on the same train/test split from Rayyan:
  1. Persistence baseline        — PCI_NEXT_pred = PCI_SCORE (no fitting)
  2. Pavement-age/trend baseline — linear regression on PAVEMENT_AGE_YEARS alone
  3. Linear Regression           — full feature set, scaled
  4. Random Forest Regression    — full feature set
  5. Gradient Boosting Regression— full feature set
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from src import results_io

# Luke's feature set. Drawn from config so it stays in step with the rest of the
# project; the extra context columns below are ones he used that config classes as
# context rather than features.
NUMERIC_FEATURES = [c for c in config.NUMERIC_FEATURE_COLS]
CATEGORICAL_FEATURES = list(config.CATEGORICAL_FEATURE_COLS) + [
    "PAVEMENT_FAMILY_EXP", "FUNC_CLASS_EXP", "LTPP_DIR_EXP", "EXPERIMENT_NO_EXP",
]


def run():
    df = pd.read_csv(config.MODEL_DATASET_PATH, low_memory=False)

    train_df = df[df[config.SPLIT_COL] == config.TRAIN_VALUE].copy()
    test_df = df[df[config.SPLIT_COL] == config.TEST_VALUE].copy()
    assert set(train_df[config.PHYSICAL_ROAD_ID_COL]).isdisjoint(
        set(test_df[config.PHYSICAL_ROAD_ID_COL])
    ), "Physical road leakage between train and test!"

    target = config.TARGET_COL
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    features = numeric + categorical

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    # 1. persistence
    pred_persistence = test_df["PCI_SCORE"].values

    # 2. age / trend
    age_train, age_test = train_df[["PAVEMENT_AGE_YEARS"]], test_df[["PAVEMENT_AGE_YEARS"]]
    age_impute = SimpleImputer(strategy="median").fit(age_train)
    trend = LinearRegression().fit(age_impute.transform(age_train), y_train)
    pred_trend = trend.predict(age_impute.transform(age_test))

    # 3. linear regression
    lr_prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    lr = Pipeline([("prep", lr_prep), ("model", LinearRegression())]).fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    # 4. random forest
    rf = Pipeline([("prep", preprocess),
                   ("model", RandomForestRegressor(n_estimators=300,
                                                   random_state=config.RANDOM_SEED,
                                                   n_jobs=-1))]).fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    # 5. gradient boosting
    gb = Pipeline([("prep", preprocess),
                   ("model", GradientBoostingRegressor(
                       random_state=config.RANDOM_SEED))]).fit(X_train, y_train)
    pred_gb = gb.predict(X_test)

    approaches = {
        "Persistence baseline": pred_persistence,
        "Pavement-age/trend baseline": pred_trend,
        "Linear Regression": pred_lr,
        "Random Forest": pred_rf,
        "Gradient Boosting": pred_gb,
    }

    import joblib
    for name, fitted in [("linear_regression", lr), ("random_forest", rf),
                          ("gradient_boosting", gb), ("age_trend", trend)]:
        joblib.dump(fitted, os.path.join(config.MODELS_DIR, f"{name}.joblib"))

    rows = [{"model": name, **results_io.score(y_test, pred)}
            for name, pred in approaches.items()]
    results_io.append_results(rows)

    preds = test_df[[config.ROAD_ID_COL, config.PHYSICAL_ROAD_ID_COL,
                     config.SURVEY_DATE_COL, "PCI_SCORE"]].copy()
    preds["PCI_NEXT_ACTUAL"] = y_test.values
    for name, pred in approaches.items():
        preds["PRED_" + name.upper().replace(" ", "_").replace("/", "_")] = pred
    preds.to_csv(config.RESULTS_PREDICTIONS_PATH, index=False)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    lims = (0, 100)
    for ax, (name, pred) in zip(axes, approaches.items()):
        s = results_io.score(y_test, pred)
        ax.scatter(y_test, pred, alpha=0.5, s=20, edgecolor="none")
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
        ax.set_title(f"{name}\nMAE={s['mae']:.2f}  RMSE={s['rmse']:.2f}  R2={s['r2']:.3f}")
        ax.set_xlabel(f"Actual {target}"); ax.set_ylabel(f"Predicted {target}")
        ax.set_xlim(lims); ax.set_ylim(lims); ax.legend(loc="upper left", fontsize=8)
    axes[-1].axis("off")
    fig.suptitle(f"Actual vs. Predicted {target} — test set (n={len(y_test)})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(config.PLOTS_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close(fig)

    print(pd.DataFrame(rows).sort_values("rmse").to_string(index=False))
    return rows


if __name__ == "__main__":
    run()
