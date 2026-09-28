"""
results_io.py — shared, idempotent read/write helpers for every stage that
produces model results (owner: Ravtej, per Decision Log #6 in v1, extended
for v2).

=== v2 CHANGE — why this file exists now ===
In v1 there was one results file (model_performance.csv) shared by Luke's
and Abdullah's stages, keyed only on "model" name. With two targets that
design would silently collide: a "Random Forest" row trained on PCI_NEXT
and a "Random Forest" row trained on RUT_NEXT would overwrite each other
under the old dedup key, and nobody would notice until the numbers looked
wrong. Fixed by giving each target its OWN results file
(config.TARGETS[key]["results_metrics_path"]) — the separation is
structural (different files), not dependent on every stage remembering to
also tag rows with a target column. Use the functions below rather than
reading/writing those CSVs directly, so re-running a stage updates a
model's row instead of duplicating it.

Contract for Luke's and Abdullah's regression stages (baseline_traditional.py,
xgboost_shap.py), one call per target:
    results_io.append_results(config.TARGETS[key]["results_metrics_path"],
                               [{"model": "Random Forest", "mae": ..., "rmse": ..., "r2": ...}, ...])

Contract for Abdullah's PCI-below-55 classifier (repair_classifier.py):
    results_io.append_classifier_results(config.CLASSIFIER["results_metrics_path"],
        [{"model": "Logistic Regression", "precision": ..., "recall": ..., "f1": ...,
          "tp": ..., "fp": ..., "fn": ..., "tn": ...}, ...])
"""
import os

import pandas as pd

REGRESSION_COLUMNS = ["model", "mae", "rmse", "r2"]
CLASSIFIER_COLUMNS = ["model", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]


def _append(path: str, rows: list, required_columns: list, dedup_key: str = "model") -> pd.DataFrame:
    if not rows:
        raise ValueError("append called with no rows.")

    new_df = pd.DataFrame(rows)
    missing = set(required_columns) - set(new_df.columns)
    if missing:
        raise ValueError(f"Row(s) missing required column(s) {missing}; expected {required_columns}.")
    new_df = new_df[required_columns]

    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing = existing[~existing[dedup_key].isin(new_df[dedup_key])]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(path, index=False)
    return combined


def append_results(path: str, rows: list) -> pd.DataFrame:
    """Append/replace regression rows (model, mae, rmse, r2). Re-running a
    model overwrites its own row rather than duplicating it."""
    return _append(path, rows, REGRESSION_COLUMNS)


def append_classifier_results(path: str, rows: list) -> pd.DataFrame:
    """Append/replace classifier rows (model, precision, recall, f1, tp, fp, fn, tn)."""
    return _append(path, rows, CLASSIFIER_COLUMNS)


def read_results(path: str):
    """Return the results DataFrame, or None if the stage hasn't run yet."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def combined_comparison(targets: dict) -> "pd.DataFrame | None":
    """v2 NEW — used by evaluate.py to build the 'both targets' comparison
    table the v2 allocation doc asks for. `targets` is config.TARGETS.
    Returns one DataFrame with a 'target' column prepended, or None if
    neither target has results yet.
    """
    frames = []
    for key, spec in targets.items():
        df = read_results(spec["results_metrics_path"])
        if df is not None:
            df = df.copy()
            df.insert(0, "target", spec["label"])
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)
