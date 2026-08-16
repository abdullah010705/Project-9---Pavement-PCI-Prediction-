"""
run_pipeline.py — single entry point for the whole project (owner: Ravtej).

    python run_pipeline.py                # use cached data/model_dataset.csv
    python run_pipeline.py --rebuild-dataset   # regenerate it from raw data first
    python run_pipeline.py --stop-on-missing   # error instead of skipping unfinished stages

Runs, in order: dataset construction -> schema validation -> EDA -> modelling ->
evaluation -> cross-validation. Every stage reads its inputs and
writes its outputs through config.py, so everyone is guaranteed to be using
the same dataset, features, target, and train/test split.

Stages that haven't been written yet (src/eda.py, src/models/*.py) raise
NotImplementedError from a run() stub — by default the pipeline prints a
skip notice and moves on, so the pipeline is runnable end-to-end at every
point in the project, not just once all five scripts exist. Pass
--stop-on-missing once everyone's code has landed, to make an unimplemented
stage a hard failure instead of a skip.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src import schema_checks


def run_dataset_stage(rebuild: bool) -> None:
    if rebuild or not os.path.exists(config.MODEL_DATASET_PATH):
        print(f"[pipeline] Building {config.MODEL_DATASET_PATH} from {config.RAW_DATA_PATH} ...")
        from src.data import build_dataset  # imported lazily: needs scikit-learn, only when rebuilding
        build_dataset.build_model_dataset()
    else:
        print(f"[pipeline] Using cached {config.MODEL_DATASET_PATH} "
              f"(pass --rebuild-dataset to regenerate from raw data).")


def run_stage(name: str, owner: str, run_fn, stop_on_missing: bool) -> bool:
    """Run one stage's run() function. Returns True if it ran, False if skipped."""
    print(f"\n[pipeline] --- {name} (owner: {owner}) ---")
    start = time.time()
    try:
        run_fn()
    except NotImplementedError as e:
        if stop_on_missing:
            raise
        print(f"[pipeline] SKIPPED — {e}")
        return False
    else:
        print(f"[pipeline] {name} done in {time.time() - start:.1f}s")
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild-dataset", action="store_true",
                         help="Regenerate model_dataset.csv from raw data instead of using the cached copy.")
    parser.add_argument("--stop-on-missing", action="store_true",
                         help="Treat an unimplemented stage as a hard failure instead of skipping it.")
    parser.add_argument("--skip-cv", action="store_true",
                         help="Skip the cross-validation stage (the slowest part).")
    parser.add_argument("--fast", action="store_true",
                         help="Cut the hyperparameter search right down. For demos and "
                              "smoke tests, not for results you intend to report.")
    args = parser.parse_args()

    if args.fast:
        os.environ["PCI_SEARCH_ITER"] = "6"
        print("[pipeline] --fast: hyperparameter search reduced to 6 candidates. "
              "Do not report these numbers.")

    print("[pipeline] === Pavement PCI prediction pipeline ===")

    # 1. Dataset construction (Rayyan)
    print("\n[pipeline] --- dataset construction (owner: Rayyan) ---")
    run_dataset_stage(args.rebuild_dataset)

    # 2. Schema validation (Ravtej) — fail fast before wasting time on later stages
    print("\n[pipeline] --- schema validation (owner: Ravtej) ---")
    df = schema_checks.load_and_validate()
    print(f"[pipeline] model_dataset.csv OK: {df.shape[0]} rows x {df.shape[1]} cols")

    # 3. EDA (Aum)
    from src import eda
    run_stage("EDA", "Aum", eda.run, args.stop_on_missing)

    # 4. Modelling (Luke + Abdullah)
    from src.models import baseline_traditional, xgboost_shap
    run_stage("baseline + traditional ML", "Luke", baseline_traditional.run, args.stop_on_missing)
    run_stage("XGBoost + SHAP", "Abdullah", xgboost_shap.run, args.stop_on_missing)

    # 5. Evaluation (Ravtej — aggregates Luke's and Abdullah's results)
    from src import evaluate
    run_stage("evaluation", "Ravtej", evaluate.run, args.stop_on_missing)

    # 6. Cross-validation over the whole dataset — the held-out split is only 182
    #    rows, too few to separate the models. Skip with --skip-cv if in a hurry.
    if not args.skip_cv:
        from src import cross_validate
        run_stage("cross-validation", "Abdullah", cross_validate.run, args.stop_on_missing)
    else:
        print("\n[pipeline] --- cross-validation SKIPPED (--skip-cv) ---")

    print("\n[pipeline] === Done ===")


if __name__ == "__main__":
    main()
