# Provenance — where each file came from

Who wrote what, and what was changed during assembly. For results, how to run it, and
open issues, see `README.md`. For task-by-task delivery against the allocation
document, see `ALLOCATION_STATUS.md`.

| Stage | File | Source | Status |
|---|---|---|---|
| Dataset construction | `src/data/build_dataset.py` | Rayyan's `build_model_dataset.py` | **assembled — Rayyan to review** |
| Schema validation | `src/schema_checks.py` | Aum's folder (labelled Ravtej) | as delivered |
| EDA | `src/eda.py` | Aum | as delivered |
| Baselines + traditional ML | `src/models/baseline_traditional.py` | Luke's `pci_prediction_pipeline.py` | **assembled — Luke to review** |
| XGBoost + SHAP | `src/models/xgboost_shap.py` | Abdullah | authored |
| Cross-validation | `src/cross_validate.py` | Abdullah | authored |
| Evaluation | `src/evaluate.py` | Aum's folder (labelled Ravtej) | as delivered |
| Orchestrator, config | `run_pipeline.py`, `config.py` | Aum's folder (labelled Ravtej) | as delivered |
| Results helper | `src/results_io.py` | Abdullah | authored |

## What changed during assembly

**Rayyan's and Luke's modules were adapted, not rewritten.** Logic, seeds and approaches
are unchanged. What changed: each is wrapped in a `run()` function, reads `config.py`
instead of hardcoded paths and column lists, and writes results through
`src/results_io.py`. Both should review and take ownership — per the allocation document
this integration work is Ravtej's remit.

**Three files are new:**

- `src/results_io.py` — `src/evaluate.py` expects one results table with columns
  `model, mae, rmse, r2`. This lets both modelling stages append to it without
  overwriting each other, and makes re-runs idempotent rather than duplicating rows.
- `src/cross_validate.py` — see below.
- Model persistence — both modelling stages now save fitted models to `models/` as
  `.joblib`, so a demo can predict without retraining.

## Why the cross-validation stage was added

Every model was being scored on the 182-row held-out split. That is correct practice,
but 182 rows cannot resolve differences of about 1 RMSE point — bootstrap intervals for
XGBoost vs Gradient Boosting cross zero, and the ordering flips between runs.

Re-scoring over all 976 rows with grouped 5-fold CV changed two conclusions:

1. The horizon variant is clearly the best model (16.95 vs 17.65 RMSE), where on the
   small split it appeared worse.
2. Linear Regression does **not** beat the persistence baseline. It appeared to on the
   held-out split (21.30 vs 23.95) and does not over the full dataset (22.42 vs 21.56).

Neither is a criticism of anyone's work — the held-out split is what the allocation
document specified. It is the reason the stage exists.

## Superseded folders

`Project9 - Merged Pipeline` was an intermediate assembly step and is now fully
contained in this folder. Safe to delete.
