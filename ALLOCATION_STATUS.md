# Allocation status — every task from Project_Group_Allocations, and where it lives

Traceability from the allocation document to the code that implements it.

---

## Rayyan — Temporal dataset construction
`src/data/build_dataset.py` · run via `python run_pipeline.py --rebuild-dataset`

| Allocated task | Status |
|---|---|
| Identify each unique road using STATE_CODE, SHRP_ID, CONSTRUCTION_NO | done — `ROAD_ID`, `PHYSICAL_ROAD_ID` |
| Sort observations by survey date | done |
| Detect rebuild events so lives aren't merged | done — `CONSTRUCTION_NO` + age-reset safety net |
| One chronological history per section | done — `PAVEMENT_SEGMENT_ID` |
| Select agreed input variables | done — declared in `config.py` |
| Lagged / cumulative variables | done — `PCI_PREV`, `ESAL_SINCE_LAST_SURVEY`, etc. |
| Create `PCI_NEXT` target | done |
| Drop rows with no future survey | done — 597 dropped |
| Leak-safe train/test grouping | done — GroupShuffleSplit on physical road, verified disjoint |
| Export `model_dataset.csv` | done — 976 rows × 54 cols |

**Finished when: everyone has a single modelling dataset — met.**

## Ravtej — Integration & pipeline
`run_pipeline.py`, `config.py`, `src/schema_checks.py`, `src/evaluate.py`

| Allocated task | Status |
|---|---|
| Shared GitHub repository | **not done** — no repo exists yet; this folder is ready to push |
| Folder structure (`/data /src /models /results /plots`) | done |
| `requirements.txt` | done |
| `config.py` with paths, features, target, seed | done |
| Dataset/schema checks before the pipeline runs | done — `src/schema_checks.py` |
| Master `run_pipeline.py` | done |
| Call members' scripts in the right order | done — verified end to end |
| Same inputs/target/split for everyone | done — enforced through `config.py` |
| Resolve naming/format integration problems | done — `src/results_io.py` gives one results schema |
| Integrate outputs into the final demonstration | not started |

**Finished when: project runs from one entry point — met.**

## Aum — EDA & data-quality analysis
`src/eda.py`

| Allocated task | Status |
|---|---|
| Dataset info: observations, roads, variable types, descriptives | done |
| Missing values | done |
| Unusual/extreme values | done — IQR flags |
| PCI distribution plot | done |
| PCI vs age / traffic / temperature / precipitation / thickness | done |
| Correlation matrix | done |
| Identify variables related to future PCI | done |
| Save plots to `/plots` automatically | done — 16 figures |
| Short summary of findings for the report | done — `results/eda_summary.md` |

**Finished when: team understands the dataset — met.** Also independently found the
PCI rut defect.

## Luke — Baseline + traditional ML models
`src/models/baseline_traditional.py`

| Allocated task | Status |
|---|---|
| Load Rayyan's exact train/test data | done — reads `SPLIT`, does not re-split |
| Persistence baseline | done |
| Pavement-age/trend baseline | done |
| Linear Regression | done |
| Random Forest | done |
| Gradient Boosting | done |
| Same train/test for every model | done |
| MAE, RMSE, R² | done |
| Save predictions | done — `results/predictions.csv` |
| Save performance to a common results file | done — `results/model_performance.csv` |
| Actual-vs-predicted plots | done — `plots/actual_vs_predicted.png` |
| Compare whether ML beats the baselines | done — **but see the correction below** |

**Finished when: clear performance table — met.**

## Abdullah — Advanced ML + XAI
`src/models/xgboost_shap.py`, `src/cross_validate.py`

| Allocated task | Status |
|---|---|
| XGBoost on the same train/test data as Luke | done |
| Initial XGBoost performance | done |
| Tuning code: trees, learning rate, depth, subsampling, regularisation | done — all five, randomised search |
| Select best config by cross-validation | done — grouped CV on training rows only |
| MAE, RMSE, R² | done |
| Compare XGBoost against Luke's models | done — plus bootstrap intervals |
| SHAP analysis | done |
| Global feature importance | done |
| SHAP summary plot | done |
| Individual prediction explanations | done — in `Abdullah - XGBoost + SHAP/` |
| Identify factors influencing deterioration | done |
| Save SHAP figures/results for report and LLM stage | done — `results/shap_values.csv` |

**Finished when: know whether XGBoost beats the simpler models, and can explain why —
met, with the caveat that all figures move once the PCI defect is fixed.**

---

## Correction to an earlier conclusion

Luke's README states Linear Regression beats the persistence baseline on RMSE and R².
That held on the 182-row test split (21.30 vs 23.95). It does **not** hold under grouped
cross-validation over all 976 rows:

| Model | RMSE (held-out, n=182) | RMSE (grouped CV, n=976) |
|---|---|---|
| Linear Regression | 21.30 | 22.42 |
| Persistence baseline | 23.95 | 21.56 |

Over the full dataset, Linear Regression is *worse* than persistence, and the bootstrap
says the difference is not distinguishable from zero. The 182-row result was noise.

This is not a criticism of Luke's work — his instructions were to use the held-out split,
which he did correctly. It is the reason `src/cross_validate.py` was added.

## Overall workflow status

1. Rayyan → modelling dataset — **done**
2. Aum → understand/visualise dataset — **done**
3. Luke + Abdullah → train and compare models — **done**
4. Ravtej → integrate into one functioning system — **done here; GitHub repo outstanding**

## Not in the allocation document, but required by the charter

- Literature review of pavement deterioration modelling and AI applications — **not started**
- Thesis report — **not started**
- Oral presentation and demonstration — **not started**
- LLM interpretation layer (stretch, Week 11) — **not started**
- Fixing the PCI rut defect — **blocking every accuracy figure above**
