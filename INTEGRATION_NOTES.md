# v2 integrated project — what this is, and what needs checking

Everyone's v2 work in one runnable project. Integrated by Abdullah because Review 2 is
close; **integration is Ravtej's allocation and he should take it back.**

```bash
./run_pipeline.py            # or: PCI_SEARCH_ITER=15 python run_pipeline.py
```

Runs: dataset check → schema validation → EDA → feature reduction → baselines +
traditional ML (both targets) → XGBoost + SHAP (both targets) → repair classifier →
comparison vs baselines → evaluation. Verified end to end.

---

## Where each piece came from

| Stage | File | Source | Status |
|---|---|---|---|
| Dataset build | `src/data/build_dataset.py` | Rayyan v2 | as delivered |
| Config, pipeline, schema checks, evaluate | `config.py`, `run_pipeline.py`, `src/schema_checks.py`, `src/evaluate.py` | Ravtej v2 | as delivered |
| EDA | `src/eda.py` | Aum v2 | as delivered + a `run()` wrapper |
| Baselines + traditional ML | `src/models/baseline_traditional.py` | **reimplemented** | **Luke must verify** |
| Feature reduction | `src/models/feature_reduction.py` | Abdullah | new |
| XGBoost + SHAP | `src/models/xgboost_shap.py` | Abdullah | new |
| Repair classifier | `src/models/repair_classifier.py` | Abdullah | new |
| Comparison vs baselines | `src/models/compare_vs_baselines.py` | Abdullah | new |

### Changes made during integration

- **Aum's `eda.py`**: added a `run()` function that calls his `main()` with the pipeline's
  configured paths. His analysis code is untouched.
- **`config.py`**: `REDUCED_FEATURE_COLS` populated with the reduced sets (was `None`).
- **`run_pipeline.py`**: two stages wired in — feature reduction before modelling (it
  writes the feature list the models read), comparison after.
- **`results/model_performance.csv`**: placeholder rows cleared. It contained
  `Persistence 8.4, RF 5.9, XGBoost 5.2`, which are not real figures.

---

## Results

### PCI_NEXT — test set, 182 surveys

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 9.50 | **13.20** | **0.762** |
| Linear Regression | 10.18 | 14.22 | 0.724 |
| XGBoost (tuned) | 9.93 | 14.30 | 0.720 |
| Gradient Boosting | 10.65 | 14.58 | 0.709 |
| Age trend (anchored) | 9.11 | 14.61 | 0.708 |
| Persistence baseline | 9.34 | 14.83 | 0.699 |
| Age trend (absolute) | 21.50 | 26.92 | 0.009 |

### RUT_NEXT — test set, 129 surveys

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Gradient Boosting | 1.09 | **1.52** | **0.811** |
| XGBoost (tuned) | 1.01 | 1.54 | 0.806 |
| Persistence baseline | 0.98 | 1.64 | 0.782 |
| Age trend (anchored) | 1.01 | 1.65 | 0.777 |
| Random Forest | 1.16 | 1.73 | 0.756 |
| Linear Regression | 1.69 | 3.16 | 0.184 |

Grouped CV over every row (more reliable than the small test split):
XGBoost PCI R² 0.758 / RMSE 12.54; XGBoost RUT R² 0.861 / RMSE 1.49.

### Repair classifier (PCI < 55)

Accuracy 0.934, precision 0.909, recall 0.769, F1 0.833, ROC-AUC 0.928.
Confusion matrix: TN 140, FP 3, FN 9, TP 30 — of 39 roads needing repair, 30 caught.
Always predicting "no repair" scores 78.6%, so quote precision and recall, not accuracy.

---

## The one thing that must be checked before Review 2

**Luke's script was not in the repo, so his stage was reimplemented from the method
section of his write-up.** Most of it reproduces his figures closely:

| Model | Luke reports | This implementation |
|---|---|---|
| Persistence | R² 0.699, RMSE 14.83 | R² 0.699, RMSE 14.83 — **exact match** |
| Age trend (absolute) | R² −0.013 | R² 0.009 |
| Random Forest | R² 0.776 | R² 0.762 |
| Gradient Boosting | R² 0.748 | R² 0.709 |
| Linear Regression | R² 0.687 | R² 0.724 |
| **Age trend (anchored)** | **R² 0.802, RMSE 12.05** | **R² 0.708, RMSE 14.61** |

Persistence matching to four decimal places confirms the data, split and metric code are
identical. So the anchored-trend gap is entirely in **how the curve is anchored**, not in
the setup.

This matters because Luke's headline conclusion — *no ML model beats the best baseline* —
rests on that 0.802. With the anchored trend at 0.708, the ordering reverses and Random
Forest and XGBoost both beat every baseline on PCI.

**Do not present either conclusion until Luke's code is in the repo and the two
implementations are reconciled.** That is the single highest-priority item.

---

## Still outstanding

- Luke's actual script, to settle the above.
- Ravtej to review the integration and take ownership back.
- `PCI_SEARCH_ITER` defaults to 40; runs here used 15 for speed. Use 40+ for final numbers.
- Confirm the repair-classifier operating threshold with the client
  (`plots/repair_threshold_sweep.png` shows the trade-off).
