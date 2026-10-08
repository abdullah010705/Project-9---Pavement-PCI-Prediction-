"""
run_pipeline.py — single entry point for the whole project (owner: Ravtej).

    python run_pipeline.py                # use cached data/model_dataset_{pci,rut}.csv
    python run_pipeline.py --rebuild-dataset   # regenerate both datasets + split from raw data first
    python run_pipeline.py --stop-on-missing   # error instead of skipping unfinished stages

Runs, in order: dataset construction -> schema validation (both datasets +
shared split) -> EDA -> modelling, once per target -> repair classifier ->
evaluation. Every stage reads its inputs and writes its outputs through
config.py, so everyone is guaranteed to be using the same datasets,
features, targets, and train/test split.

=== v2 CHANGE (Project Group Allocations v2) ===
PCI and rutting are now two separate targets (PCI_NEXT, RUT_NEXT), each
with its own dataset file, sharing one train/test split. This file now
loops the modelling stages once per target in config.TARGETS, instead of
calling each stage once. Luke's and Abdullah's run() functions take a
target_key argument ("pci" or "rut") and are expected to train/evaluate
against that target only — the looping lives here, not inside their code,
so their modelling logic itself stays target-agnostic (per the v2 doc:
"Model code (Luke and Abdullah) — target-agnostic, points at whatever
column we name"). The PCI-below-55 classifier (Abdullah) is single-target
(PCI only) and is run once, not looped.

Stages that haven't been written yet (src/eda.py, src/models/*.py) raise
NotImplementedError from a run() stub — by default the pipeline prints a
skip notice and moves on, so the pipeline is runnable end-to-end at every
point in the project, not just once all scripts exist. Pass
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
    datasets_exist = all(os.path.exists(t["dataset_path"]) for t in config.TARGETS.values()) \
        and os.path.exists(config.SPLIT_ASSIGNMENT_PATH)
    if rebuild or not datasets_exist:
        print(f"[pipeline] Building both modelling datasets from {config.RAW_DATA_PATH} ...")
        from src.data import build_dataset  # imported lazily: needs scikit-learn, only when rebuilding
        build_dataset.build_model_datasets()
    else:
        print(f"[pipeline] Using cached {[t['dataset_path'] for t in config.TARGETS.values()]} "
              f"(pass --rebuild-dataset to regenerate from raw data).")


def run_stage(name: str, owner: str, run_fn, stop_on_missing: bool) -> bool:
    """Run one stage's run() function (optionally with args baked in via a
    lambda/partial). Returns True if it ran, False if skipped."""
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
                         help="Regenerate both model_dataset_{pci,rut}.csv + split_assignment.csv from raw data.")
    parser.add_argument("--with-llm", action="store_true",
                         help="Also generate sample condition briefs with the LLM layer "
                              "(Component E). Needs Ollama for a real brief; falls back to "
                              "the deterministic template if Ollama is unavailable.")
    parser.add_argument("--stop-on-missing", action="store_true",
                         help="Treat an unimplemented stage as a hard failure instead of skipping it.")
    args = parser.parse_args()

    print("[pipeline] === Pavement condition prediction pipeline (v2: PCI + rutting, separate targets) ===")

    # 1. Dataset construction (Rayyan) — builds both datasets + the shared split in one pass
    print("\n[pipeline] --- dataset construction (owner: Rayyan) ---")
    run_dataset_stage(args.rebuild_dataset)

    # 2. Schema validation (Ravtej) — both datasets + the shared-split cross-check.
    #    Fails fast, before wasting time on later stages, if either dataset or the split drifts.
    print("\n[pipeline] --- schema validation (owner: Ravtej) ---")
    dfs = schema_checks.load_and_validate_all()
    for key, df in dfs.items():
        print(f"[pipeline] {config.TARGETS[key]['dataset_path']} OK: {df.shape[0]} rows x {df.shape[1]} cols")

    # 3. EDA (Aum) — one call; eda.run() is expected to cover both datasets internally
    #    (including the "do PCI and rutting move together" comparison — see src/eda.py contract)
    from src import eda
    run_stage("EDA (both datasets)", "Aum", eda.run, args.stop_on_missing)

    # 3b. Feature reduction (Abdullah) — must run BEFORE modelling: it writes
    #     results/reduced_features_<key>.json, which the model stages read.
    from src.models import feature_reduction
    run_stage("feature reduction", "Abdullah", feature_reduction.run, args.stop_on_missing)

    # 4. Modelling (Luke + Abdullah) — run once per target, per the v2 allocation doc
    from src.models import baseline_traditional, xgboost_shap
    for key, spec in config.TARGETS.items():
        run_stage(f"baseline + traditional ML — {spec['label']}", "Luke",
                   lambda k=key: baseline_traditional.run(k), args.stop_on_missing)
        run_stage(f"XGBoost + SHAP — {spec['label']}", "Abdullah",
                   lambda k=key: xgboost_shap.run(k), args.stop_on_missing)

    # 4b. PCI-below-55 repair classifier (Abdullah) — single target, not looped
    from src.models import repair_classifier
    run_stage("repair classifier (PCI < 55)", "Abdullah", repair_classifier.run, args.stop_on_missing)

    # 5. Evaluation (Ravtej — aggregates Luke's and Abdullah's results, both targets)
    from src.models import compare_vs_baselines
    run_stage("comparison vs baselines", "Abdullah", compare_vs_baselines.run,
              args.stop_on_missing)

    # 5c. LLM condition briefs (Luke, Component E) — optional, off by default.
    #     Generates plain-English briefs from the model outputs. The LLM never produces
    #     a number: llm/payload.py builds the facts and llm/validate.py checks every
    #     figure in the text against them.
    if args.with_llm:
        from src import llm_stage
        run_stage("LLM condition briefs", "Luke", llm_stage.run, args.stop_on_missing)

    from src import evaluate
    run_stage("evaluation (both targets)", "Ravtej", evaluate.run, args.stop_on_missing)

    print("\n[pipeline] === Done ===")


if __name__ == "__main__":
    main()
