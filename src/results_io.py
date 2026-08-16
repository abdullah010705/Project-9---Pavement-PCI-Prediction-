"""
results_io.py — shared helper so every modelling stage writes results the same way.

src/evaluate.py (Ravtej) expects config.RESULTS_METRICS_PATH to contain one row per
model with columns exactly: model, mae, rmse, r2. Stages append to it rather than
overwriting, so Luke's and Abdullah's results end up in one table.

Appending naively means re-running the pipeline duplicates every row, so
append_results() removes any existing rows for the same model names first. That makes
each stage idempotent: run the pipeline five times, still one row per model.
"""
import os

import numpy as np
import pandas as pd

import config

COLUMNS = ["model", "mae", "rmse", "r2"]


def score(y_true, y_pred) -> dict:
    """MAE, RMSE, R2 — the three metrics the allocation document asks for."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def append_results(rows) -> pd.DataFrame:
    """Append result rows to config.RESULTS_METRICS_PATH, replacing same-named rows.

    rows: list of dicts with keys model, mae, rmse, r2
    """
    new = pd.DataFrame(rows)
    missing = set(COLUMNS) - set(new.columns)
    if missing:
        raise ValueError(f"result rows are missing column(s): {sorted(missing)}")
    new = new[COLUMNS]

    path = config.RESULTS_METRICS_PATH
    if os.path.exists(path):
        existing = pd.read_csv(path)
        if set(COLUMNS).issubset(existing.columns):
            existing = existing[~existing["model"].isin(new["model"])]
            new = pd.concat([existing[COLUMNS], new], ignore_index=True)

    new.to_csv(path, index=False)
    return new
