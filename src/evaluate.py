"""
evaluate.py — evaluation stage (owner: Ravtej — aggregates Luke's and
Abdullah's results into one comparison, rather than each of them building
their own comparison logic against the other's output).

=== v2 CHANGE ===
Now aggregates across BOTH targets (PCI_NEXT, RUT_NEXT), each read from its
own results file (config.TARGETS[key]["results_metrics_path"] — see
results_io.py for why they're separate files rather than one shared file).
Produces a per-target comparison chart (as in v1) plus one combined table
covering both targets, which is what the v2 "Finished when" line for
Luke/Abdullah asks for ("a performance table covering both targets").

Does not touch the PCI-below-55 classifier (config.CLASSIFIER) — precision/
recall/F1/confusion-matrix don't belong on the same MAE/RMSE/R2 table.
Abdullah's repair_classifier.py stage reports those separately.

Contract Luke's and Abdullah's stages must follow for this to work: for
each target, call
    results_io.append_results(config.TARGETS[key]["results_metrics_path"], rows)
with rows shaped {"model": ..., "mae": ..., "rmse": ..., "r2": ...}. Model
names should be human-readable, e.g. "Persistence baseline", "Random
Forest", "XGBoost (tuned)".
"""
import matplotlib.pyplot as plt

import config
import results_io


def _plot_target(target_key: str, results):
    spec = config.TARGETS[target_key]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ["mae", "rmse", "r2"]):
        ax.barh(results["model"], results[metric])
        ax.set_title(f"{spec['label']} — {metric.upper()}")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(spec["comparison_plot_path"], dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved {spec['label']} comparison chart to {spec['comparison_plot_path']}")


def _report_target(target_key: str):
    spec = config.TARGETS[target_key]
    results = results_io.read_results(spec["results_metrics_path"])
    if results is None:
        print(f"[evaluate] {spec['results_metrics_path']} does not exist yet "
              f"(Luke's and Abdullah's {spec['label']} stages write it) — skipping.")
        return None

    required = {"model", "mae", "rmse", "r2"}
    missing = required - set(results.columns)
    if missing:
        print(f"[evaluate] {spec['results_metrics_path']} is missing column(s) {missing} "
              f"— expected {sorted(required)}. Skipping.")
        return None

    results = results.sort_values("rmse")
    print(f"\n[evaluate] {spec['label']} model comparison (sorted by RMSE, lower is better):")
    print(results.to_string(index=False))

    baseline_names = results["model"].str.contains("baseline|persistence", case=False, na=False)
    if baseline_names.any() and (~baseline_names).any():
        best_baseline_rmse = results.loc[baseline_names, "rmse"].min()
        best_model_row = results.loc[~baseline_names].sort_values("rmse").iloc[0]
        improvement = 100 * (best_baseline_rmse - best_model_row["rmse"]) / best_baseline_rmse
        if improvement > 0:
            print(f"[evaluate] {spec['label']}: best non-baseline model ('{best_model_row['model']}') "
                  f"beats the best baseline by {improvement:.1f}% RMSE.")
        else:
            print(f"[evaluate] {spec['label']}: WARNING — best non-baseline model does NOT beat "
                  f"the best baseline.")

    _plot_target(target_key, results)
    return results


def run():
    per_target = {}
    for key in config.TARGETS:
        result = _report_target(key)
        if result is not None:
            per_target[key] = result

    combined = results_io.combined_comparison(config.TARGETS)
    if combined is not None:
        print("\n[evaluate] Combined comparison across both targets:")
        print(combined.sort_values(["target", "rmse"]).to_string(index=False))
    else:
        print("\n[evaluate] Neither target has results yet — nothing to combine.")

    return per_target or None


if __name__ == "__main__":
    run()
