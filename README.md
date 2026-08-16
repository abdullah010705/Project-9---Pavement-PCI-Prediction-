# Pavement PCI Prediction — Project 9

Predicts a road's condition at its **next** survey (`PCI_NEXT`) from its current
condition, age, traffic loading, climate, structure and maintenance history, using LTPP
pavement data.

All five members' code, assembled per `Project_Group_Allocations`, running from one
entry point. See `ALLOCATION_STATUS.md` for task-by-task traceability.

---

## Run it

```bash
./run.sh
```

That is the whole thing. On first run it builds a virtual environment and installs
dependencies (a couple of minutes); after that it just runs, in about 100 seconds.

Needs Python 3.10 or newer — macOS ships 3.9, which will not work. `run.sh` checks and
tells you if that is the problem. On macOS XGBoost also needs Apple's OpenMP runtime
(`brew install libomp`); `run.sh` checks that too and says so rather than failing with a
library error.

If you would rather manage the environment yourself:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

Flags pass straight through, e.g. `./run.sh --skip-cv`.

| Flag | Effect |
|---|---|
| `--rebuild-dataset` | Regenerate `model_dataset.csv` from raw data instead of using the cached copy |
| `--fast` | Cut the hyperparameter search to 6 candidates. Whole pipeline in ~20 seconds. **Demo only — do not report these numbers** |
| `--skip-cv` | Skip cross-validation |
| `--stop-on-missing` | Fail on an unimplemented stage rather than skipping it (currently passes) |

### Runtime

Almost all the time is the hyperparameter search in the XGBoost stage. Two environment
variables control it, so you never have to edit code:

```bash
PCI_SEARCH_ITER=60 ./run.sh     # thorough, for the final thesis numbers
PCI_SEARCH_ITER=6  ./run.sh     # same as --fast
PCI_N_JOBS=2       ./run.sh     # if the machine is struggling
```

`PCI_N_JOBS` defaults to 4. It is deliberately **not** `-1`: with only 794 training rows
the threading overhead exceeds the work per tree, and on a machine with many cores
letting XGBoost take all of them made this stage roughly 18x slower.

## What happens when you run it

| # | Stage | Owner | What it does |
|---|---|---|---|
| 1 | Dataset construction | Rayyan | Builds the modelling table: one row per survey, paired with the PCI at that road's next survey. Splits pavement histories at rebuild events. Assigns a leak-safe train/test split grouped by physical road. |
| 2 | Schema validation | Ravtej | Checks the dataset matches `config.py` before anything downstream runs. Fails loudly and early. |
| 3 | EDA | Aum | Descriptives, missingness, outliers, correlations, 16 figures, written summary. |
| 4 | Baselines + traditional ML | Luke | Persistence, age-trend, Linear Regression, Random Forest, Gradient Boosting. |
| 5 | XGBoost + SHAP | Abdullah | Tuned XGBoost with grouped CV on training rows only, then SHAP explanations. |
| 6 | Evaluation | Ravtej | Aggregates every model into one comparison table and chart. |
| 7 | Cross-validation | Abdullah | Re-scores everything over all 976 rows, because 182 test rows cannot separate the models. |

## Layout

```
config.py            paths, feature groups, target, seed — import this, never hardcode
run_pipeline.py      the single entry point
data/                master_with_pci_2.csv (raw), model_dataset.csv (built)
src/
  data/build_dataset.py            dataset construction   (Rayyan)
  schema_checks.py                 validation             (Ravtej)
  eda.py                           EDA                    (Aum)
  models/baseline_traditional.py   baselines + standard ML (Luke)
  models/xgboost_shap.py           XGBoost + SHAP         (Abdullah)
  evaluate.py                      results aggregation    (Ravtej)
  cross_validate.py                grouped CV             (Abdullah)
  results_io.py                    shared results schema
models/              fitted models (.joblib)
results/             metrics, predictions, SHAP values, EDA tables
plots/               all figures
```

**Everyone imports `config.py`.** Paths, column lists, the target and the seed live
there. If the schema changes, change that file, not seven scripts.

---

## Results

Two evaluations. The held-out split is what the allocation document asked for; the
cross-validation is the more reliable comparison and was added because the held-out set
is too small to separate models.

### Grouped 5-fold cross-validation — all 976 rows

| Model | MAE | RMSE | R² | Beats persistence? |
|---|---|---|---|---|
| XGBoost (tuned + horizon) | **12.39** | **16.95** | **0.609** | yes, CI [3.53, 5.78] |
| XGBoost (tuned) | 12.92 | 17.65 | 0.576 | yes, CI [2.87, 5.01] |
| Gradient Boosting | 13.22 | 18.01 | 0.559 | yes, CI [2.51, 4.66] |
| Random Forest | 12.82 | 18.06 | 0.556 | yes, CI [2.48, 4.56] |
| Persistence baseline | 13.51 | 21.56 | 0.367 | — |
| Linear Regression | 17.26 | 22.42 | 0.316 | no |

### Held-out test set — 182 rows

| Model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost (tuned) | 13.15 | 17.57 | 0.634 |
| XGBoost (tuned + horizon) | 13.21 | 17.79 | 0.625 |
| Gradient Boosting | 13.46 | 18.52 | 0.593 |
| Random Forest | 14.46 | 19.70 | 0.540 |
| Linear Regression | 16.51 | 21.30 | 0.462 |
| Persistence baseline | 15.13 | 23.95 | 0.320 |
| Pavement-age/trend baseline | 23.79 | 28.87 | 0.012 |

### What holds, and what doesn't

**Holds:** the tree models beat the persistence baseline decisively, by 3.5–4.6 RMSE
points with confidence intervals well clear of zero. Persistence is a hard baseline —
"the road will be the same next year" — so beating it by ~21% is the project's main
result so far.

**Holds:** pavement age alone is nearly useless (R² 0.012), which justifies the
multi-factor approach over simple age-based curves.

**Does not hold:** the ordering of individual models on the 182-row split. XGBoost and
Gradient Boosting differ by less than the noise there, and the two XGBoost variants swap
places between the two evaluations. Quote the cross-validation numbers, not the held-out
ones, for any model-vs-model claim.

**Does not hold:** Linear Regression beating persistence. It appeared to on the held-out
split and does not over the full dataset. See `ALLOCATION_STATUS.md`.

---

## Two open issues

**1. The PCI target is defective for ~20% of rows.** `pci_algorithm.py` applies no
rutting deduct when rut depth is missing, so PCI is computed on two different scales.
Aum's EDA and Abdullah's SHAP analysis found this independently — 90% of large apparent
PCI *improvements* coincide with a missing next-survey rut measurement, against a 17%
base rate. `RUT_DEPTH_1_8IN__isna` is currently the 2nd most important feature in the
XGBoost model: it is learning how the label was built, not how pavements deteriorate.

Every number above changes once this is fixed. Full write-up in
`Abdullah - XGBoost + SHAP/PCI_TARGET_DEFECT.md`.

**2. The forecast horizon is disputed.** `config.py` marks `YEARS_TO_NEXT_SURVEY` as
"must NOT be used as a model input". Under cross-validation, including it is the single
biggest improvement available (16.95 vs 17.65 RMSE). The argument for including it: in
deployment you *choose* the horizon, so it is known at prediction time. Both variants
are reported rather than overriding the shared config. The team should pick one.

## Known limitations

- 976 rows, 259 roads, GPS-1/GPS-2 asphalt sections only.
- Hyperparameters are tuned once on the training split, then held fixed during
  cross-validation. Re-tuning inside each fold would be more rigorous.
- `N_SEARCH_ITER = 30` in `xgboost_shap.py` keeps the pipeline quick — raise it for the
  final thesis run.
- The PCI values themselves use approximated ASTM deduct curves, documented in
  `PCI_Methodology_and_Assumptions.md`. They are not official ASTM scores.
