# CLAUDE.md — Handover: ZEIT3101 Pavement Deterioration Project (V0.0.3) → LLM workstream

> **Integration note (added when the LLM layer was merged into the pipeline).**
> `llm/` now lives *inside* the pipeline folder rather than beside a separate `V0.0.3/`.
> `v002.py` detects either layout by looking for `config.py`, and `V002_DIR` still
> overrides it, so nothing else changed. `run_llm.sh` is the macOS/Linux equivalent of
> `run_llm.cmd` / `run_llm.ps1`, and `./run.sh --with-llm` runs the briefs as an optional
> pipeline stage. See the project README for details.



**Owner of this workspace:** Luke (ADFA, Mech Eng, ZEIT3101 IT Project 2, Group 9)
**Handover date:** 24 Sep 2026
**Codebase state:** `V0.0.3/` — the v2 integrated pipeline, last touched 20 Sep 2026 (integrated by Abdullah)
**Your job in this folder:** help Luke build an LLM component on top of the existing ML pipeline, and help close out Luke's own open items in that pipeline.
**Last updated:** 30 Sep 2026. The LLM layer (Component E) is built in `llm/`. See **§10** for the current state, and `llm-plan.md` for the plan and evaluation log.

Read this whole file before touching anything. Section 7 (known inconsistencies) matters more than it looks. Section 10 is where the LLM work stands now.

---

## 0. Working rules

1. **Australian English** in all prose, comments and docs (analyse, colour, modelling, behaviour).
2. **Metric units** in anything Luke writes up. The data is in LTPP (US) units — see §3.4 for conversions. Do not rewrite source data; convert at the presentation layer.
3. **`V0.0.3/` is the team's shared code.** Do not edit files owned by other members (see §2 table) without Luke's say-so. Build the LLM work in a **new sibling folder** (suggested: `AAA. LLM/llm/`) that *imports* from `V0.0.3`, rather than modifying it.
4. **Never touch `data/`** except via `--rebuild-dataset`. Never re-split. The train/test split is `SPLIT` column + `data/split_assignment.csv`, grouped by `PHYSICAL_ROAD_ID`. Anything that leaks test rows into training, tuning or feature selection is a defect.
5. **Use `config.py` for paths/columns and `results_io.py` for writing results.** No hardcoded paths or column names.
6. **Numbers come from files, not from docs.** The README, INTEGRATION_NOTES and TUNING_CHANGE_NOTE all quote figures that no longer match `results/` (see §7). The CSV/JSON files in `results/` are the only source of truth, and even those are from a demo-speed run.
7. **The LLM must never generate a number.** Every figure in LLM output must be retrieved from a results file or computed by code and passed in. This is the central design constraint for the LLM component (see §8).
8. Ask Luke before any decision that changes a reported number, a team-owned file, or project scope.

---

## 1. Project context

**Unit:** ZEIT3101 IT Project 2 (UNSW Canberra at ADFA).
**Title:** *Can Artificial Intelligence Predict Road Failures? A Data-Driven Study of Pavement Deterioration.*
**Data:** US FHWA LTPP (Long-Term Pavement Performance) survey data, GPS-1/GPS-2 asphalt sections only.

**Objectives (from the project outline):**
1. Review pavement deterioration modelling approaches.
2. Pre-process pavement condition datasets.
3. Develop and compare ML models for deterioration prediction.
4. **Investigate the role of AI tools, including LLMs, in supporting pavement asset-management decisions.** ← this workstream.

The outline specifically names two LLM uses: **(a) interpreting maintenance records** and **(b) generating explainable insights from prediction results.**

**Deliverables:** curated dataset, ML models, thesis report, oral presentation + live demo of the prediction framework.

**Client decisions already made (do not reopen):**
- PCI and rutting are separate targets, not one merged score (Decision Log #12).
- Repair threshold = **PCI_NEXT < 55** (client-confirmed).
- HDM-4/Austroads empirical benchmark dropped; Luke's age-trend baseline (power-law/exponential form) is the comparison instead (Decision Log #13). Must be noted in the thesis limitations.
- ~120 features is too many; reduced sets are required.

---

## 2. Team and ownership

| Member | Owns | Files |
|---|---|---|
| **Rayyan** | Dataset construction, PCI calculator v2 | `src/data/build_dataset.py`, `src/data/pci_algorithm_v2_reference.py` |
| **Aum** | EDA | `src/eda.py` |
| **Luke** | Baselines + traditional ML; original author of `pci_algorithm.py` | `src/models/baseline_traditional.py` (**currently a reimplementation, not Luke's code**) |
| **Abdullah** | Feature reduction, XGBoost + SHAP, repair classifier, baseline comparison | `src/models/{feature_reduction,xgboost_shap,repair_classifier,compare_vs_baselines,reconcile_baseline}.py` |
| **Ravtej** | Config, pipeline, schema checks, evaluation, integration, Decision Log, lock file | `config.py`, `run_pipeline.py`, `src/schema_checks.py`, `src/evaluate.py`, `results_io.py`, `requirements-lock.txt`, `Decision_Log_v2.docx` |

Integration is formally Ravtej's; Abdullah did it for V0.0.3 because Review 2 was close. It is meant to go back to Ravtej.

---

## 3. Data

### 3.1 Files (`V0.0.3/data/`)

| File | Rows × cols | Role |
|---|---|---|
| `master_with_pci_3.csv` | 12,255 × 148 | **Raw v2 input.** LTPP survey records with PCI recomputed distress-only. `PCI_SCORE_OLD` = v1 score (with rutting). |
| `model_dataset_pci.csv` | 976 × 55 | Built. Target `PCI_NEXT`, plus `NEEDS_REPAIR`. 259 physical roads, 327 pavement lives. |
| `model_dataset_rut.csv` | 693 × 54 | Built. Target `RUT_NEXT`. 207 roads, 257 lives. |
| `split_assignment.csv` | 259 × 2 | `PHYSICAL_ROAD_ID → train/test`. Source of truth for the split. |
| `master_with_pci_2.csv`, `model_dataset.csv` | — | **v1 legacy.** Do not use. |
| `README_model_dataset.md` | — | v1 dataset methodology (road identity, lags) — still valid for v2. |

### 3.2 Row semantics
One row = one survey of one pavement life, paired with the condition at that life's **next** survey.
- `PHYSICAL_ROAD_ID` = `STATE_CODE-SHRP_ID` (location; never resets). **Grouping key for all splits/CV.**
- `ROAD_ID` = `STATE_CODE-SHRP_ID-CONSTRUCTION_NO` (one pavement life; resets at rebuild).
- Lag features (`*_PREV`, `*_CHANGE_SINCE_LAST`, `DAYS_SINCE_LAST_SURVEY`) are NaN on the first survey of each life — structural, not error.
- Test split: 182 rows (PCI), 129 rows (RUT). Too small to separate models reliably — grouped CV over all rows is the more trustworthy comparison.

### 3.3 Feature groups (defined in `config.py`)
Age · Traffic (ESAL) · Climate (7 cols) · Structure (thickness, layers, materials) · Maintenance (`PRIOR_PATCH_COUNT`, `DAYS_SINCE_LAST_PATCH`) · Current condition (`PCI_SCORE`, `RUT_DEPTH_1_8IN`, IRI, MRI, `PCI_RATING`, `PCI_DOMINANT_DISTRESS`) · Target-specific lags. Iterate `config.TARGETS` (`"pci"`, `"rut"`); the v1 single-target aliases are deprecated.

### 3.4 Units (convert for anything Luke presents)
| Column | Source unit | Metric |
|---|---|---|
| `RUT_DEPTH_1_8IN`, `RUT_NEXT`, `RUT_PREV` | eighths of an inch | × 3.175 → mm |
| `IRI_*`, `MRI` | LTPP standard is m/km — **verify against raw values before labelling** | — |
| `*_THICKNESS_MM` | mm | already metric |
| `CUM_ESAL_AT_SURVEY` | equivalent single axle loads (80 kN) | dimensionless count |
| Climate cols | verify units in LTPP CLM tables before labelling | — |

### 3.5 What the data does NOT contain (important for the LLM brief)
There are **no free-text maintenance records** anywhere in the dataset. "Maintenance history" is only `PRIOR_PATCH_COUNT`, `LAST_PATCH_DATE` (raw file), `DAYS_SINCE_LAST_PATCH` (~93% missing) and the per-survey patch/pothole distress quantities. Outline use (a) — *interpreting maintenance records* — has no text corpus to work on unless one is sourced (LTPP MNT/RHB tables have coded work types, not narratives) or synthetic records are generated and clearly labelled as such. Raise this with Luke before designing anything for (a).

---

## 4. Pipeline

### 4.1 Stage order (`run_pipeline.py`)
1. Dataset (cached unless `--rebuild-dataset`)
2. Schema validation — both datasets + shared-split cross-check (`schema_checks.load_and_validate_all()`)
3. EDA (Aum)
4. Feature reduction (Abdullah) → writes `results/reduced_features_<key>.json`
5. Per target: baselines + traditional ML (Luke) → XGBoost + SHAP (Abdullah)
6. Repair classifier, PCI only (Abdullah)
7. Comparison vs baselines — test split + grouped CV + paired bootstrap (Abdullah)
8. Evaluation — combined table (Ravtej)

Stages raising `NotImplementedError` are skipped unless `--stop-on-missing`.

### 4.2 Running it on Luke's machine (Windows) — **the shipped `.venv` will not work**
`V0.0.3/.venv` was created on Abdullah's Mac (`/Users/abdullah/...`, Homebrew Python 3.12.13, `bin/` layout). It is useless on Windows and should be deleted/ignored. `run.sh` is bash and Mac-oriented.

On Windows (PowerShell), from `V0.0.3/`:
```powershell
py -3.12 -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PCI_SEARCH_ITER = "10"; python run_pipeline.py      # demo speed
$env:PCI_SEARCH_ITER = "40"; python run_pipeline.py      # anything quoted in the report
```
Python ≥ 3.10 required. `.vscode/launch.json` hardcodes `.venv/bin/python` (Mac path) — Windows needs `.venv-win\Scripts\python.exe`. Delete stale `__pycache__/` folders (built under 3.10 and 3.12).

**As built on Luke's machine (30 Sep 2026):**
- **Venv.** It lives at `%USERPROFILE%\.venvs\zeit3101` (Python 3.12.10), **outside OneDrive**, so package files don't sync. `.venv-win` is not used. Pins are in `llm/requirements.txt`: pandas 2.3.3 and numpy 2.2.6 as in the lock file, plus xgboost 3.4.1, shap 0.52.0, scikit-learn 1.9.1 and joblib 1.6.0. `run_llm.cmd` creates it and installs into it.
- **The shipped `models/*.joblib` files do NOT load with `joblib.load` on Windows** ("input stream corrupted" / "Invalid cast, from Null to Object"). Only the learner *config* block is incompatible; the trained model is intact. `llm/v002.booster()` loads just the model section. **Never refit to work around this.** The loaded models reproduce `results/` exactly: PCI test RMSE 14.160, rutting 1.627, classifier TN 134 / FP 9 / FN 8 / TP 31. `tests/` and `llm/check_env.py` check this. Tell Abdullah.
- **The pipeline has NOT been re-run here.** `run_pipeline.py` overwrites the shared `results/` and `models/`. Ask Luke before running it in place, or run it on a copy.
- **In PowerShell, call Windows PowerShell by its full path** (`$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe`), because `powershell` is not on PATH in the tool shell. Python output with `→` or `—` crashes the cp1252 console unless `PYTHONIOENCODING=utf-8` is set; `run_llm.ps1` sets it.

**Reproducibility caveat:** `requirements-lock.txt` only has pandas/numpy (and matplotlib/seaborn) truly pinned; scikit-learn, xgboost, shap, joblib are `[NOT VERIFIED]` minimums (Decision Log #18). Different library versions produce different numbers. Once Luke's environment runs cleanly, `pip freeze` it and offer the pins to Ravtej.

### 4.3 `PCI_SEARCH_ITER`
Code default is **25** (`src/models/xgboost_shap.py:63`). README and INTEGRATION_NOTES say 40. The shipped `results/` came from a fast run. **No number currently in `results/` is report-grade.**

---

## 5. Current results — as actually on disk (`V0.0.3/results/`, 20 Sep 2026, fast run)

### PCI_NEXT — test, n = 182 (`model_performance_pci.csv`)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 9.53 | **13.21** | **0.761** |
| XGBoost (tuned) | 9.89 | 14.16 | 0.726 |
| Linear Regression | 10.18 | 14.22 | 0.724 |
| Gradient Boosting | 10.64 | 14.57 | 0.710 |
| Age trend (anchored) — *reimplementation* | 9.11 | 14.61 | 0.708 |
| Persistence | 9.34 | 14.83 | 0.699 |
| XGBoost (default) | 10.87 | 15.52 | 0.671 |
| Age trend (absolute) | 21.50 | 26.92 | 0.009 |

Grouped CV, all rows: XGBoost RMSE 12.27 / R² 0.768 vs anchored trend 13.24 / 0.730 vs persistence 13.46 / 0.721. Paired bootstrap: **XGBoost better than anchored trend** (ΔRMSE 0.96, 95% CI 0.49–1.41).

### RUT_NEXT — test, n = 129 (`model_performance_rut.csv`)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Gradient Boosting | 1.09 | **1.52** | **0.812** |
| XGBoost (tuned) | 1.13 | 1.63 | 0.784 |
| Persistence | 0.98 | 1.64 | 0.782 |
| Age trend (anchored) | 1.01 | 1.65 | 0.777 |
| Random Forest | 1.16 | 1.73 | 0.756 |
| Linear Regression | 1.18 | 1.70 | 0.763 |
| XGBoost (default) | 1.20 | 1.78 | 0.741 |
| Age trend (absolute) | 2.85 | 3.65 | −0.089 |

(Units: eighths of an inch. RMSE 1.52 ≈ 4.8 mm.) Bootstrap: XGBoost vs anchored trend **not distinguishable** on rutting.

### Repair classifier, PCI < 55 (`repair_classifier_summary.json`)
Test: accuracy 0.907, precision 0.775, recall 0.795, F1 0.785, ROC-AUC 0.926. TN 134 / FP 9 / FN 8 / TP 31 (31 of 39 repair cases caught). Majority-class accuracy 0.786 — quote precision/recall, not accuracy. Operating threshold still to be confirmed with the client (`plots/repair_threshold_sweep.png`).

### SHAP (PCI, `shap_values_pci.csv`)
Dominated by current condition: `PCI_SCORE` mean |SHAP| 13.8, then `PCI_RATING_Good` 2.6, `DAYS_SINCE_LAST_SURVEY` 1.2, `SUBBASE_THICKNESS_MM` 1.1, `BASE_THICKNESS_MM` 0.9. Per-row explanations of four showcase cases in `individual_explanations_*.csv` + `plots/xgboost_case_*.png`.

---

## 6. Luke's open items in the pipeline (highest priority first)

### 6.1 Anchored age-trend baseline: 0.802 vs 0.708 — **blocks the project's headline claim**
- Luke's write-up (19 Sep) reports anchored trend on PCI at **R² 0.802 / RMSE 12.05**. At that figure, no ML model beats the best baseline.
- The reimplementation scores **0.708 / 14.61**. At that figure, RF and XGBoost both beat every baseline on PCI.
- Persistence matches Luke's figure to four decimals → data, split and metric code are identical. The gap is purely in how the curve is built/anchored.
- **Luke's original script is lost.** It cannot be diffed.
- `src/models/reconcile_baseline.py` tested six variants (`results/reconcile_baseline_pci.csv`): best was **D, linear-in-age curve: R² 0.755 / RMSE 13.39**. None reach 0.802. README's line "none reaches 0.802" is correct but omits that D closes half the gap.

**Recommended path:** reconstruct from Luke's memory + write-up, then test the untried hypotheses on identical rows:
1. Per-road deterioration rate (each life's own `PCI_CHANGE_SINCE_LAST / YEARS_SINCE_LAST_SURVEY`, extrapolated over `YEARS_TO_NEXT_SURVEY`, falling back to the population curve on first surveys).
2. Exponential form `PCI = a·exp(−b·age)` (the form Decision #13 names) rather than log-linear.
3. Curve fitted on (age + gap) → `PCI_NEXT` directly.
4. Curve fitted per stratum (e.g. by `PCI_RATING`, climate zone, or base type).
5. Clip/threshold differences, and any use of `PCI_PREV`.
6. Accidental test-row leakage of a kind not in variant B (e.g. fitting per road including test surveys).

If nothing reproduces 0.802 in reasonable time, the defensible position is: report the best **reproducible** anchored baseline, state in the thesis that an earlier figure of 0.802 could not be reproduced, and let the grouped-CV + bootstrap results decide the headline. Don't let the team quote either conclusion until this is settled.

### 6.2 Verify `baseline_traditional.py`
It reimplements Luke's six approaches from his method section. Reproduction vs Luke's figures: Persistence exact; RF 0.762 vs 0.776; GB 0.709 vs 0.748; LR 0.724 vs 0.687; absolute trend 0.009 vs −0.013. Luke needs to confirm hyperparameters (RF 500 trees, GB 300 trees depth 3), preprocessing (median impute, one-hot, scaling only for LR), and whether he used the full or reduced feature set. **Note:** this stage uses the full feature set and ignores `reduced_features_*.json`, unlike XGBoost — so the traditional-vs-XGBoost comparison is not on like-for-like inputs.

### 6.3 `pci_algorithm.py` reconciliation (Decision Log #15, blocking)
Two uploads overwrote each other; Rayyan built v2 from one copy. Client asked for the rutting-removal change to be made with the original author (Luke). Luke must diff his copy against `src/data/pci_algorithm_v2_reference.py`. If they differ, `master_with_pci_3.csv` must be regenerated and the pipeline re-run with `--rebuild-dataset`. Also note: the deduct curves in that file are an **approximation**, not digitised ASTM D6433 Appendix X3 curves — a thesis limitation.

---

## 7. Known inconsistencies in V0.0.3 (documentation drift — fix or flag before Review 2)

| # | Where | Says | Actually (on disk) |
|---|---|---|---|
| 1 | README / INTEGRATION_NOTES | XGBoost PCI 0.720 / 14.30 | 0.726 / 14.16 |
| 2 | TUNING_CHANGE_NOTE | XGBoost PCI 0.748 / 13.583 after tuning fix | 0.726 / 14.16 (likely a different `PCI_SEARCH_ITER` run) |
| 3 | README / INTEGRATION_NOTES | XGBoost RUT 0.806 / 1.54 | 0.784 / 1.63 |
| 4 | README / INTEGRATION_NOTES | Classifier acc 0.934, P 0.909, R 0.769, F1 0.833, TN140/FP3/FN9/TP30 | acc 0.907, P 0.775, R 0.795, F1 0.785, TN134/FP9/FN8/TP31 |
| 5 | README, config.py comment | PCI reduced to **10** features, CV RMSE 12.708 → 12.418 | `reduced_features_pci.json`: **15** features, 12.729 → 12.336 |
| 6 | `config.REDUCED_FEATURE_COLS` | Hand-copied list | **Not read by any model.** Models read the JSON. The two have drifted (JSON has `PCI_PREV`, `FREEZE_THAW`, `EVAPORATION`, `PRECIPITATION`; config doesn't). |
| 7 | README / INTEGRATION_NOTES | `PCI_SEARCH_ITER` defaults to 40 | Code default 25 |
| 8 | README | "Rutting unaffected" by baseline issue; GB best on RUT | Correct — but XGBoost RUT now trails persistence on MAE |
| 9 | EDA summary | PCI "one clear dominant peak" | Decision Log #16: right-skewed with ceiling spike at 100 — qualify it |
| 10 | Feature reduction | Presented as a fixed result | Selection is not stable run-to-run (10 → 15 features). Needs a fixed seed audit or a stability analysis |

Root cause for 1–7: prose was written from one run, then the pipeline was re-run. Fix: one full `PCI_SEARCH_ITER=40` run on a pinned environment, then regenerate all quoted numbers from `results/` programmatically (a small script that writes the README results block would stop this recurring — offer it to Ravtej).

Also noteworthy for SHAP/LLM work: `individual_explanations_pci.csv` has a "largest overestimate" case with actual PCI 0.0, predicted 68.6. Actual-zero rows next to high current PCI may be data artefacts (rebuild not captured by `CONSTRUCTION_NO`, or a PCI calc failure — check `PCI_ERROR` in the raw file). Worth checking before an LLM narrates that case as a "real" failure.

---

## 8. The LLM workstream

### 8.1 Scope — DECIDED (30 Sep 2026)
Luke chose: **both outline uses, (a) maintenance record interpretation and (b) explainable insights**, on **Ollama (local) + Python**. The plan is `llm-plan.md`, where (b) is **Component E** and (a) is **Component M**. Component E is built (§10). Component M is blocked until Luke picks a data source: M1 (LTPP coded MNT/RHB records), M2 (clearly labelled synthetic records) or M3 (hybrid, recommended). The brief says the LLM work runs "depending on data availability", so it is an extension, not a core deliverable. Keep it off the critical path of Objectives 1–3.

Original candidate directions (kept for context):

**A. Explainable insight generation (strongest fit — outline use (b)).**
Input: one road-survey row + model prediction(s) + per-row SHAP values + repair-classifier probability. Output: a short plain-English condition brief for an asset manager — predicted next PCI / rut depth with uncertainty, repair flag, the three or four drivers behind it, and a recommended action tier. All numbers injected from code; the LLM only writes language around them.

**B. Project Q&A / results assistant.**
RAG over `results/*.csv|json`, `eda_v2_summary.md`, Decision Log, README. Answers "which model is best for rutting and why?", "what are the open issues?". Useful for the demo and for the team; weaker as research contribution.

**C. Maintenance record interpretation (outline use (a)).**
Blocked by §3.5 — no text records exist. Options: source LTPP maintenance/rehab tables and have the LLM map coded work types to plain descriptions and link them to condition jumps; or use a small, clearly labelled synthetic set to demonstrate the capability. Needs a scope decision with the supervisor/client.

**D. Hybrid (likely best for the thesis):** A as the core, with a lightweight B for the demo.

### 8.2 Design constraints (non-negotiable)
- **Grounding:** the LLM receives a structured JSON payload of facts; it may not state any figure not in the payload. Validate outputs programmatically (regex every number in the output against the payload; reject on mismatch).
- **Traceability:** every generated brief stores its input payload, prompt version, model name and timestamp.
- **Honest uncertainty:** include the model's test/CV error (e.g. PCI RMSE ≈ 12–14 points) so the brief does not overclaim. Rutting improvements over persistence are not statistically distinguishable — the LLM must not say otherwise.
- **Direction of SHAP:** SHAP explains the model, not the pavement. Wording must be "the model's prediction was pulled down mainly by…", not "the road will fail because…".
- **Units:** present rut depth in mm, keep PCI on 0–100.
- **Read-only against V0.0.3:** load models from `V0.0.3/models/*.joblib` and data via `config` / `schema_checks`; do not retrain inside the LLM code.
- **Evaluation plan from day one:** factual-consistency check (automated), plus a small human rating rubric (accuracy, usefulness, clarity) over a fixed sample of test-set roads (e.g. 20 stratified by PCI band and repair flag). This is what turns the LLM work into a thesis result rather than a demo.

### 8.3 Open questions to put to Luke
1. ~~Which direction?~~ Decided: both (a) and (b) (§8.1).
2. ~~Which LLM?~~ Decided: local Ollama, so there is no API cost and no data leaves the machine. Primary model is **not yet confirmed**: the evidence favours `qwen3:8b` (§10.4). `run_llm.ps1` defaults to it, but `llm_client.DEFAULT_MODEL` is still `llama3.1:8b` until Luke confirms.
3. Is this Luke's individual allocation or a team deliverable? Who else consumes it? *(open)*
4. Demo format: CLI, Streamlit, or a static HTML report per road? *(open; a CLI exists)*
5. Dates for Review 2 and the final demo? *(open)*
6. Component M data source: M1, M2 or M3? *(open, blocks Component M)*
7. Action-tier rules in `llm/payload.py` are provisional. Confirm them with the client. *(open)*
8. May the V0.0.3 pipeline be re-run in place? *(open)*

### 8.4 Suggested first session — DONE
Environment, model loading, `payload.py`, prompts, validator, evaluation and launcher are all built. See §10. Keep §6.1 (anchored baseline) moving; it still gates what the whole team can claim.

---

## 9. File map (`V0.0.3/`)

```
config.py                 paths, columns, TARGETS, CLASSIFIER, seed, REPAIR_THRESHOLD=55
run_pipeline.py           single entry point
run.sh                    bash launcher (Mac-oriented)
results_io.py             idempotent results writers
requirements.txt          minimum versions (what run.sh installs)
requirements-lock.txt     partially verified pins
Decision_Log_v2.docx      decisions #1–18; #9, #15, #16, #18 open
README.md, INTEGRATION_NOTES.md, TUNING_CHANGE_NOTE.md   — see §7 for drift
src/data/build_dataset.py, pci_algorithm_v2_reference.py
src/eda.py, schema_checks.py, evaluate.py
src/models/baseline_traditional.py (reimpl.), feature_reduction.py, xgboost_shap.py,
           repair_classifier.py, compare_vs_baselines.py, reconcile_baseline.py
models/   xgboost_pci.joblib, xgboost_rut.joblib, repair_classifier.joblib
results/  metrics, predictions, SHAP, OOF, feature sweeps, EDA tables, reconcile output
plots/    EDA, SHAP, case explanations, comparisons, confusion matrix, threshold sweep
.venv/    Abdullah's macOS venv — ignore on Windows
```

Open Decision Log items: **#9** include `YEARS_TO_NEXT_SURVEY` as a feature? (improves CV RMSE; horizon is known at prediction time — team to decide) · **#15** pci_algorithm reconciliation · **#16** EDA distribution wording · **#18** lock file.

---

## 10. LLM layer — current state (30 Sep 2026)

### 10.1 Workspace layout
```
AAA. LLM/
  CLAUDE.md, llm-plan.md    this file; the plan (v0.4) with milestones and evaluation log (§10a)
  run_llm.ps1, run_llm.cmd  launcher: checks → installs → starts Ollama → runs (never writes V0.0.3)
  V0.0.3/                   team code, read-only
  llm/                      Component E (see llm/README.md for the full map)
    v002.py                 the ONLY module that touches V0.0.3 (data, feature matrix, model loading, results)
    payload.py              fact payload per ROAD_ID + SURVEY_DATE; for_prompt() is all the LLM sees
    features_lookup.py      labels + display units (rut ×3.175 → mm; unverified units are not shown)
    schema.py               BriefText output fields (Ollama JSON schema) + markdown renderer
    validate.py             grounding (numbers), completeness, meaning (driver direction, repair flag), wording
    template_brief.py       deterministic brief = fallback + evaluation baseline
    brief.py                prompt registry (PROMPTS), Ollama call, validate → retry once → template fallback, trace
    llm_client.py, cli.py, check_env.py
    prompts/                brief_v1.j2 + brief_v1_example.json; brief_v2.j2 + brief_v2_examples.json (default)
    eval/                   sample.csv (fixed 20 test roads), run_eval.py, rubric.md, results/
    outputs/briefs.jsonl    trace of every brief generated via the CLI
  tests/test_llm_layer.py   23 tests, no Ollama needed
```

### 10.2 How to run (from `AAA. LLM/`)
```powershell
.\run_llm.cmd check                                    # all requirement checks
.\run_llm.cmd brief --road 53-1006-3 --date 2003-04-16 --show-hindsight
.\run_llm.cmd -NoLLM brief --road ... --date ...       # template only
.\run_llm.cmd test
$py = "$HOME\.venvs\zeit3101\Scripts\python.exe"
& $py -m llm.eval.run_eval run --models qwen3:8b llama3.1:8b --prompt brief_v2 --no-template
& $py -m llm.eval.run_eval compare     # paired v1-vs-v2 on the same roads, exact McNemar
& $py -m llm.eval.run_eval rescore     # re-validate saved briefs with the current validator
```
Survey dates are accepted as `yyyy-mm-dd` or the dataset's `d/m/yyyy`.

### 10.3 Conventions — follow these when extending
- **Prompt versions are immutable.** To change wording or examples, add `brief_vN.j2` and its examples file, and register it in `PROMPTS` in `brief.py`. Never edit an old version, because traces record the prompt name and a hash of its files.
- **Worked examples** must be **training-split** roads and must pass `validate()`. The tests check both. Freeze each example's payload view into the JSON.
- **Validator changes:** add a test, check the template brief still passes on all 183 test surveys (a test does this), then run `run_eval rescore` so earlier runs are re-scored on the same checks.
- **The evaluation sample is fixed** (`llm/eval/sample.csv`: test split, one survey per road, stratified by PCI band × repair flag, seed 42). Don't redraw it: `make_sample()` refuses if the file exists.
- **Hindsight**, the observed next-survey values, is in the payload for evaluation only. It is never shown to the LLM, and raters should not look at it before rating.
- **Held back from the brief:** there is no claim that the PCI model beats the age-trend baseline until §6.1 is settled. The rutting brief always carries the "not reliably better than persistence" caveat, read from the bootstrap verdict in `results/`.
- **Action tiers are decided in code (`payload._action`), never by the LLM.** The current rules are provisional (§8.3 #7).
- **SHAP driver effects are measured from the model's average prediction** (`model_average` = the SHAP base value = the training mean of the target: PCI 69.5, rut 23.1 mm). The brief states this.

### 10.4 Evaluation results so far (20 test roads, temperature 0, seed 42; details in `llm-plan.md` §10a)
| Prompt | Model | First-try pass* | Accepted with a flipped driver direction | Median s/brief |
|---|---|---|---|---|
| v1 | qwen3:8b | 45% | 0 | 8.7 |
| v1 | llama3.1:8b | 40% | 1 | 20.8 |
| v2 | qwen3:8b | **100%** | 0 | 6.7 |
| v2 | llama3.1:8b | 90% | 2 | 14.2 |

\*Scored with the current validator. v1 → v2 paired improvement: p = 0.0005 (llama3.1), p = 0.001 (qwen3). Key thesis finding: **checking numbers alone is not enough**. llama3.1 reversed SHAP directions, which a sign-insensitive number check passed; the meaning checks now catch this. **Human rating has not been done yet.** The sheet is `llm/eval/results/rating_20260930-112808.csv` and the rubric is `llm/eval/rubric.md`.

### 10.5 Next steps
1. Human rating of the v2 run: at least two raters, with inter-rater agreement.
2. Confirm the primary model (qwen3:8b recommended) and the action tiers.
3. Component M, once a data source is chosen.
4. Demo front end (Streamlit suggested). Also test `phi4-mini`, because the brief assumes a "standard laptop".
5. §6.1 anchored-baseline reconstruction, which can be done read-only in a separate script.
6. Team housekeeping: tell Abdullah about the joblib/Windows issue, and offer Ravtej the pins in `llm/requirements.txt` for Decision Log #18.
