"""
evaluate.py — evaluation stage (owner: Ravtej — aggregates Luke's and
Abdullah's results into one comparison, rather than each of them building
their own comparison logic against the other's output).

Unlike eda.py / baseline_traditional.py / xgboost_shap.py, this one is not
a stub: it's real, but it can only do anything once config.RESULTS_METRICS_PATH
exists (written by Luke's and Abdullah's stages, one row per model with
columns: model, mae, rmse, r2). Until then it skips cleanly.

Contract Luke and Abdullah's stages must follow for this to work:
  - append (not overwrite) rows to config.RESULTS_METRICS_PATH
  - use exactly these column names: model, mae, rmse, r2
  - model names should be human-readable, e.g. "Persistence baseline",
    "Random Forest", "XGBoost (tuned)"
"""
import os

import pandas as pd
import matplotlib.pyplot as plt

import config


def run():
    if not os.path.exists(config.RESULTS_METRICS_PATH):
        print(f"[evaluate] {config.RESULTS_METRICS_PATH} does not exist yet "
              f"(Luke's and Abdullah's stages write it) — nothing to evaluate, skipping.")
        return None

    results = pd.read_csv(config.RESULTS_METRICS_PATH)
    required = {"model", "mae", "rmse", "r2"}
    missing = required - set(results.columns)
    if missing:
        print(f"[evaluate] {config.RESULTS_METRICS_PATH} is missing column(s) {missing} "
              f"— expected {sorted(required)}. Skipping.")
        return None

    results = results.sort_values("rmse")
    print("[evaluate] Model comparison (sorted by RMSE, lower is better):")
    print(results.to_string(index=False))

    baseline_names = results["model"].str.contains("baseline|persistence", case=False, na=False)
    if baseline_names.any() and (~baseline_names).any():
        best_baseline_rmse = results.loc[baseline_names, "rmse"].min()
        best_model_row = results.loc[~baseline_names].sort_values("rmse").iloc[0]
        improvement = 100 * (best_baseline_rmse - best_model_row["rmse"]) / best_baseline_rmse
        print(f"\n[evaluate] Best non-baseline model ('{best_model_row['model']}') beats "
              f"the best baseline by {improvement:.1f}% RMSE." if improvement > 0 else
              f"\n[evaluate] WARNING: best non-baseline model does NOT beat the best baseline.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ["mae", "rmse", "r2"]):
        ax.barh(results["model"], results[metric])
        ax.set_title(metric.upper())
        ax.invert_yaxis()
    fig.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, "model_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved comparison chart to {out_path}")

    return results


if __name__ == "__main__":
    run()
