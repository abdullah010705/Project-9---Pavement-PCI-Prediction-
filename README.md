# Pavement Condition Prediction (v2 — PCI and rutting, separate targets)

Predicts future pavement condition from current condition, traffic,
climate, structure, and maintenance history, using LTPP road survey data.

**v2 change (client-directed, after Review 1):** PCI and rutting are two
different measurements and are no longer merged into one score. There are
now two regression targets, each with its own dataset:

- `PCI_NEXT` — distress-only PCI, computed by `pci_algorithm.py` (rutting
  no longer contributes a deduct — see Decision Log #12)
- `RUT_NEXT` — rut depth at the next survey

Plus one classification target, PCI dataset only: `NEEDS_REPAIR` (1 if
`PCI_NEXT` < 55, the client's repair threshold).

## Run

```bash
./run.sh          # the modelling pipeline — nine stages
./run_llm.sh      # the LLM condition briefs (Component E)
```

**Two entry points, because they are two different things.** `./run.sh` builds the
datasets, trains and evaluates every model, and writes all the results and figures.
`./run_llm.sh` turns those saved models into plain-English briefs for individual roads.
You can also fold the briefs into the pipeline run with `./run.sh --with-llm`.

On first run either script creates `.venv` and installs what it needs. After that
`./run.sh` takes about two to three minutes.

| Command | What it does |
|---|---|
| `./run.sh` | Normal run — all nine modelling stages |
| `./run.sh --with-llm` | As above, plus three sample condition briefs into `results/llm_sample_briefs.md` |
| `./run.sh --rebuild-dataset` | Regenerate both datasets from raw data first |
| `./run.sh --stop-on-missing` | Fail on an unimplemented stage instead of skipping (currently passes) |
| `PCI_SEARCH_ITER=40 ./run.sh` | Thorough hyperparameter search — **use this for any number that goes in the report** |
| `./run_llm.sh brief --road 1-4126-1 --date 1993-03-30` | A brief for one road |
| `./run_llm.sh test` | The LLM layer's 23 tests — no Ollama needed |

`PCI_SEARCH_ITER` defaults to 40. Lower it (10 or so) for demos.

> **A fast run overwrites the reportable figures.** `results/` and `models/` are rewritten
> on every run, and the LLM layer's tests check the saved models reproduce the reported
> numbers. If you do a reduced-search run, restore the real figures afterwards with
> `PCI_SEARCH_ITER=40 ./run.sh`. The pipeline prints a warning when this applies.

**Requirements:** Python 3.10 or newer — macOS ships 3.9, which will not
work. On macOS XGBoost also needs Apple's OpenMP runtime
(`brew install libomp`). `run.sh` checks both and tells you in plain
language rather than failing with a traceback.

### If you use VS Code

**Do not use the Run/Play button before running `./run.sh` once.** The play
button launches `/usr/bin/python3` — Apple's system Python — which bypasses
the virtual environment, so every dependency appears to be missing.

`.vscode/settings.json` points VS Code at `.venv` and `.vscode/launch.json`
adds two ready configurations to the Run and Debug panel ("fast — demo
settings" and "full — for reported numbers"). Both need `.venv` to exist,
so run `./run.sh` in the terminal once first, then reload the window.

### Managing the environment yourself

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt   # pinned; see the file's header
python run_pipeline.py
```

## Pipeline order

dataset construction (both datasets + the shared split) → schema validation
(both datasets + shared-split cross-check) → EDA (both datasets) →
**feature reduction** → modelling, once per target (baseline + traditional
ML, XGBoost + SHAP) → PCI-below-55 repair classifier → **comparison vs
baselines** → evaluation (both targets).

Feature reduction runs *before* modelling because it writes
`results/reduced_features_<key>.json`, which the model stages read. If it is
absent they fall back to the full feature set, so nothing breaks — but the
numbers will be the unreduced ones.

**Finished when** (per the v2 allocation doc): one command runs the whole
project across both targets, and every team member reproduces the same
numbers from the same code.

## Repo layout

```
data/
  master_with_pci_3.csv     — raw, PCI recomputed distress-only (v2)
  model_dataset_pci.csv     — built, target PCI_NEXT (976 rows)
  model_dataset_rut.csv     — built, target RUT_NEXT (693 rows)
  split_assignment.csv      — ONE train/test split, grouped by physical
                               road, applied to both datasets by lookup
src/
  data/
    build_dataset.py        — dataset construction, both targets (Rayyan)
    pci_algorithm_v2_reference.py — Rayyan's v2 pci_algorithm.py, kept here
                               for reference; NOT wired into the pipeline
                               (PCI is pre-computed upstream, same as v1)
  eda.py                    — EDA, both datasets                (Aum)
  models/
    baseline_traditional.py — baselines + traditional ML, per target (Luke)
    feature_reduction.py     — ~130 features down to 10, evidence-led (Abdullah)
    xgboost_shap.py          — XGBoost + SHAP, per target        (Abdullah)
    repair_classifier.py     — PCI-below-55 classifier, PCI only (Abdullah)
    compare_vs_baselines.py  — head-to-head, test split + grouped CV (Abdullah)
    reconcile_baseline.py    — diagnostic for the anchored-trend discrepancy (Abdullah)
  evaluate.py                — aggregates both targets' results into one comparison
  schema_checks.py           — validates both datasets + the shared split
  results_io.py              — shared, idempotent results read/write helpers
models/       trained model artifacts (generated)
results/      metrics (per target + classifier), predictions, SHAP values (generated)
plots/        figures, per target (generated)
config.py     shared paths, column names, targets, seed — import this, don't hardcode
run_pipeline.py
run.sh        one-command launcher: builds .venv, installs, runs
.vscode/      interpreter + launch configs so the Run button uses .venv
requirements-lock.txt        pinned dependency versions (v2 new requirement)
```

## Everyone uses the same inputs

Import `config.py` rather than hardcoding column names or paths. The key
v2 structure is `config.TARGETS`, a dict keyed `"pci"` / `"rut"`:

```python
import config
for key, target in config.TARGETS.items():
    target["dataset_path"]          # which CSV to load
    target["target_col"]            # "PCI_NEXT" or "RUT_NEXT"
    target["all_feature_cols"]      # agreed model inputs for that target
    target["categorical_feature_cols"]
    target["results_metrics_path"]  # where to write MAE/RMSE/R2 rows
    target["shap_results_path"]
```

- `config.SPLIT_COL` — `"SPLIT"` (`"train"`/`"test"`), already assigned in
  both dataset files AND in `config.SPLIT_ASSIGNMENT_PATH` (the source of
  truth `schema_checks.validate_shared_split()` checks both files against)
  — don't re-split.
- `config.CLASSIFIER` — the PCI-below-55 classifier's dataset path, target
  column (`NEEDS_REPAIR`), feature list, and results path.
- `config.REPAIR_THRESHOLD` — 55, the client-confirmed repair threshold.
- `config.RESULTS_DIR`, `config.PLOTS_DIR`, `config.MODELS_DIR` — where to
  write outputs.

Before writing a new stage, load a dataset through `src/schema_checks.py`
rather than `pd.read_csv()` directly:

```python
from src import schema_checks
df = schema_checks.load_and_validate("pci")   # or "rut"
# or, to also verify the shared split across both:
dfs = schema_checks.load_and_validate_all()   # {"pci": df, "rut": df}
```

When writing results, use `results_io.py` rather than `pd.to_csv()`
directly — it dedupes by model name so re-running a stage updates its row
instead of duplicating it, and it keeps PCI's and RUT's results in
separate files so the two targets can never collide under the same model
name (see `results_io.py`'s docstring for why that matters).

## v1 backward-compatible aliases

`config.TARGET_COL`, `config.MODEL_DATASET_PATH`, `config.ALL_FEATURE_COLS`,
etc. still exist and point at the PCI target, so old scratch scripts don't
hard-crash on import. Don't write new code against them — use
`config.TARGETS` instead.


## The LLM layer (Component E, owner: Luke)

`llm/` turns a road and a survey date into a short plain-English condition brief, built
from the XGBoost predictions, their SHAP drivers and the repair classifier.

```bash
./run_llm.sh                                                   # demo brief
./run_llm.sh brief --road 1-4126-1 --date 1993-03-30
./run_llm.sh brief --road 1-4126-1 --date 1993-03-30 --no-llm  # template, no Ollama
./run_llm.sh list --split test                                 # available surveys
./run_llm.sh test                                              # 23 tests, no Ollama needed
./run_llm.sh check                                             # environment checks
./run_llm.sh eval                                              # 20-road evaluation
```

`run_llm.cmd` / `run_llm.ps1` are Luke's original Windows launchers and do the same thing.

**The LLM never produces a number.** `llm/payload.py` builds the facts in code;
`llm/validate.py` then checks every figure in the generated text against them, along with
wording, units and completeness. If validation fails it retries once with the errors, and
falls back to `llm/template_brief.py` — a deterministic brief with the same facts — if it
fails again. Every attempt is recorded in `llm/outputs/briefs.jsonl`.

A real brief needs [Ollama](https://ollama.com) installed locally **and the model
downloaded**:

```bash
brew install ollama          # if not already installed
ollama pull llama3.1:8b      # the default, ~4.7 GB
ollama pull qwen3:8b         # optional second model, used by the evaluation
```

Installing Ollama is not enough on its own — the model is a separate download, and
without it you get a 404 from the Ollama client. `run_llm.sh` checks for this and tells
you which models you do have.

Change the default with `LLM_MODEL=qwen3:8b ./run_llm.sh ...` or `--model qwen3:8b`.

Without any model, everything still runs on the template path, which is also the
evaluation baseline — but note that is deterministic code, not a language model, so it
should not be demonstrated as the LLM component.

`--with-llm` adds a pipeline stage that writes three sample briefs to
`results/llm_sample_briefs.md`. It is off by default because it is a stretch-goal
component and a real brief needs Ollama.

### Two things Luke flagged

- **Action tiers are provisional.** The corrective / preventive / investigate / monitor
  rules in `llm/payload._action()` still need confirming with the client.
- **The briefs deliberately avoid the disputed claim.** No brief says the PCI model beats
  the age-trend baseline while the 0.802 vs 0.708 disagreement is unresolved, and the
  rutting brief always states the model is not distinguishable from persistence, because
  the bootstrap result says so.

## Current results (provisional — see open items)

Held-out test split. Grouped cross-validation over every row is in
`results/comparison_vs_baselines_*.csv` and is the more reliable comparison,
since the test splits are only 182 (PCI) and 129 (RUT) surveys.

**PCI_NEXT** — Random Forest 0.762 R² / 13.20 RMSE · XGBoost (tuned) 0.720 /
14.30 · Gradient Boosting 0.709 / 14.58 · Age trend anchored 0.708 / 14.61 ·
Persistence 0.699 / 14.83

**RUT_NEXT** — Gradient Boosting 0.811 R² / 1.52 RMSE · XGBoost (tuned)
0.806 / 1.54 · Persistence 0.782 / 1.64 · Age trend anchored 0.777 / 1.65

**Repair classifier (PCI < 55)** — accuracy 0.934, precision 0.909, recall
0.769, F1 0.833. Confusion matrix TN 140 / FP 3 / FN 9 / TP 30: of 39 roads
that needed repair, 30 were caught. Always predicting "no repair" scores
78.6%, so quote precision and recall rather than accuracy.

**Feature reduction** — PCI 138 → 10 features (CV RMSE 12.708 → 12.418),
RUT 123 → 10 (1.511 → 1.502). Both targets scored *better* with 10 features
than with the full set, so this is not a trade of accuracy for simplicity.

## Known open items (not fabricated — see Decision Log for full detail)

- **Anchored age-trend baseline disagreement (blocking the headline claim).**
  Luke reports his anchored baseline at R² 0.802 / RMSE 12.05 on PCI. His
  script was not in the repo, so the stage was reimplemented from the method
  section of his write-up; that version scores 0.708 / 14.61 on the same test
  rows. Persistence matches his figure to four decimal places, so the data,
  split and metric code are identical — the difference is in how the curve is
  anchored. Six plausible variants were tested (`src/models/reconcile_baseline.py`)
  and none reaches 0.802, including the "curve fitted on all rows" leak
  hypothesis, which suggests Luke's method is genuinely better rather than
  buggy. **This decides the project's headline:** at 0.802 no ML model beats
  the best baseline; at 0.708 Random Forest and XGBoost both do on PCI. Needs
  Luke's script before either conclusion is presented. Rutting is unaffected.
- **`src/models/baseline_traditional.py` is a reimplementation, not Luke's
  code.** It runs and reproduces most of his figures, but he must verify it.
- **pci_algorithm.py reconciliation (blocking).** Rayyan's v2 delivery was
  built from a single copy of `pci_algorithm.py` after two uploads
  overwrote each other. The client asked for this change to be made "with
  whoever originally wrote the calculator" (Luke). Until Luke confirms his
  copy matches, `master_with_pci_3.csv` and everything built from it should
  be treated as provisional.
- **PCI distribution after rutting removal.** The old many-small-humps
  bimodality is gone, as predicted. The new distribution is not a single
  clean interior peak either, though — it's right-skewed with a large
  ceiling spike at PCI=100. Worth a sentence in Aum's EDA write-up rather
  than repeating "single-peaked" unqualified.
- **RUT dataset size vs the allocation doc's estimate.** Estimated ~746
  rows / 219 roads / 274 lives; actual is 693 / 207 / 257. Legitimate — the
  RUT target additionally requires a rut reading at the *next* survey, same
  rule already used for PCI. Logged in the Decision Log so it isn't a
  surprise later.
- **Task board and repository access** (both owed by Ravtej per the v2
  allocation doc) are manual account actions, not something this pipeline
  or an automated tool can do — see the chat message this repo was
  delivered with for concrete steps.
- **requirements-lock.txt** has 3 of 7 packages not yet verified against a
  real install (no network access in the integration sandbox) — see the
  file's own header. `run.sh` installs from `requirements.txt` (unpinned) for
  this reason; switch it to the lock file once the pins are confirmed.
- **Integration ownership.** The integration (wiring every member's stage into
  one runnable pipeline) is Ravtej's allocation. It was done by Abdullah
  because Review 2 was close; it should go back to Ravtej rather than
  silently transferring.
- **Placeholder results cleared.** `results/model_performance.csv` previously
  contained dummy rows (Persistence 8.4, RF 5.9, XGBoost 5.2) left over from
  testing the evaluation stage. Real PCI RMSE is around 13–15. Removed, but
  worth knowing in case they were quoted anywhere.
