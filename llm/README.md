# `llm/` — explainable condition briefs (Component E)

> **Integration note (added when the LLM layer was merged into the pipeline).**
> `llm/` now lives *inside* the pipeline folder rather than beside a separate `V0.0.4/`.
> `v002.py` detects either layout by looking for `config.py`, and `V002_DIR` still
> overrides it, so nothing else changed. `run_llm.sh` is the macOS/Linux equivalent of
> `run_llm.cmd` / `run_llm.ps1`, and `./run.sh --with-llm` runs the briefs as an optional
> pipeline stage. See the project README for details.



This package is the LLM layer on top of the V0.0.4 pipeline. It takes a road and a survey date and produces a short, plain-English condition brief. The facts in the brief come from the team's XGBoost models, their SHAP values and the repair classifier. A local Ollama model writes the sentences. See `../llm-plan.md` for the design and `../CLAUDE.md` for the project rules.

**Principle:** the LLM never produces a number. Code builds the facts (`payload.py`). Before a brief is accepted, `validate.py` checks every number in it against those facts, along with wording, units and completeness.

## Quick start: `run_llm.sh` (macOS/Linux) or `run_llm.cmd` / `run_llm.ps1` (Windows)

This one script checks requirements, installs what's missing, starts Ollama, and runs a command:

```powershell
.\run_llm.cmd                                     # demo brief (double-clicking works too)
.\run_llm.cmd check                               # checks only
.\run_llm.cmd brief --road 1-4126-1 --date 1993-03-30 --show-hindsight
.\run_llm.cmd -Model llama3.1:8b brief --road 1-4126-1 --date 1993-03-30
.\run_llm.cmd -NoLLM brief --road 1-4126-1 --date 1993-03-30   # template only, no Ollama
.\run_llm.cmd list --split test
.\run_llm.cmd test
.\run_llm.cmd eval                                # 20-road evaluation with -Model
```

The `.cmd` wrapper gets around PowerShell's script execution policy. Calling `.\run_llm.ps1` directly also works if your policy allows it. The default model is `qwen3:8b`.

What it checks, in order:
1. Python 3.12 is installed.
2. The venv exists, and is created if missing.
3. The packages in `llm/requirements.txt` are installed; missing ones are installed.
4. Ollama is installed, its server is running (started hidden if not), and the model is pulled.
5. V0.0.4 data and results files are present, all three models load, and their test RMSE matches `results/`.
6. The prompt files and the evaluation sample exist.

`python -m llm.check_env` runs checks 1, 3, 5 and 6 on their own.

## Setup (Windows, manual)

The venv lives **outside OneDrive** so thousands of package files don't sync:

```powershell
py -3.12 -m venv $HOME\.venvs\zeit3101
& $HOME\.venvs\zeit3101\Scripts\Activate.ps1
pip install -r "requirements.txt" -r "llm\requirements.txt"
ollama pull qwen3:8b; ollama pull llama3.1:8b
```

On macOS/Linux `./run_llm.sh` does all of this for you using the project's own `.venv`.

Run everything from the pipeline folder (the one containing `run_pipeline.py`).

## Use

```powershell
python -m llm.cli list --split test                          # available surveys
python -m llm.cli payload --road 1-4126-1 --date 1993-03-30  # the facts, as JSON
python -m llm.cli brief   --road 1-4126-1 --date 1993-03-30  # LLM brief (default llama3.1:8b)
python -m llm.cli brief   --road 1-4126-1 --date 1993-03-30 --model qwen3:8b --show-hindsight
python -m llm.cli brief   --road 1-4126-1 --date 1993-03-30 --no-llm      # template only
$env:LLM_MODEL = "qwen3:8b"                                  # change the default model

python -m pytest tests -q -p no:cacheprovider                # no Ollama needed
python -m llm.eval.run_eval run --models qwen3:8b llama3.1:8b
```

Every brief is appended to `llm/outputs/briefs.jsonl`. Each record holds the payload, the prompt name and hash, the model tag and digest, the Ollama version, the options, every attempt, and the validation result.

## Flow

```
payload.py ─► prompts/brief_v2.j2 + brief_v2_examples.json ─► Ollama (JSON schema, temp 0, seed 42)
          └─► validate.py ─ fail ─► retry once with the errors ─ fail ─► template_brief.py
                         └─ pass ─► schema.render_markdown ─► outputs/briefs.jsonl
```


## Execution flow — which files run, in order

Traced from a real `brief_for()` call. Fifteen files execute; everything else in the
project is untouched.

| # | File | What it does |
|---|---|---|
| 1 | `llm/__init__.py` | Package init. |
| 2 | `llm/brief.py` | **The orchestrator** — the only entry point; calls everything below. |
| 3 | `llm/llm_client.py` | Ollama wrapper; loaded even in template mode, used only if `use_llm=True`. |
| 4 | `llm/template_brief.py` | The deterministic fallback writer, prepared up front. |
| 5 | `llm/payload.py` | **Builds the facts in code** — prediction, uncertainty range, top-4 SHAP drivers, repair probability, action tier. |
| 6 | `llm/v002.py` | **The only bridge into the pipeline** — loads the dataset, rebuilds the feature matrix, loads the fitted models, reads `results/`. |
| 7 | `config.py` | Supplies pipeline paths, target definitions and feature lists. |
| 8 | `src/__init__.py` | Package init. |
| 9 | `src/data/__init__.py` | Package init. |
| 10 | `src/data/pci_algorithm_v2_reference.py` | Converts a PCI number to its rating band, and rut depth to a severity class. |
| 11 | `src/models/__init__.py` | Package init. |
| 12 | `src/models/feature_reduction.py` | `build_matrix()` — rebuilds the model inputs exactly as training did (train-only imputation, same one-hot levels). |
| 13 | `llm/features_lookup.py` | Turns column names into readable labels and converts rut depth from eighths of an inch to mm. |
| 14 | `llm/schema.py` | Defines the seven fields the LLM must fill, and renders the final Markdown. |
| 15 | `llm/validate.py` | **Checks every number in the text against the payload**, plus wording, units and completeness. |

With Ollama connected the file list is identical — `llm_client.py` simply makes a real call
between steps 5 and 14, and `validate.py` may send one retry back. The facts come from the
same code path either way.

### Two things this makes clear

**Only two pipeline files run.** `config.py` and `src/models/feature_reduction.py` (plus
the PCI reference for rating bands). The models are loaded from `models/*.joblib` — nothing
is retrained, and `xgboost_shap.py`, `baseline_traditional.py` and `repair_classifier.py`
never execute. That is why a brief returns in under a second.

**`v002.py` is the single point of contact with the pipeline.** It is the only module that
imports `config` or anything under `src/`. If the pipeline changes, that is the one file
that needs updating.

## Files

| File | Role |
|---|---|
| `interactive.py` | Asks which road and which survey, then writes the brief. Added after the client asked for direct user input; it only chooses the inputs, nothing about how a brief is produced changes. |
| `v002.py` | The only module that touches V0.0.4. It loads data, rebuilds the model input matrix exactly as `xgboost_shap.run()` does, loads the models, and reads results files. |
| `payload.py` | Builds the fact payload: predictions, uncertainty range, top-4 SHAP drivers, repair probability, action tier, and data notes. `for_prompt()` is the only view the LLM sees. |
| `features_lookup.py` | Feature labels and display units. Rut depth is converted from eighths of an inch to mm. Units that haven't been verified are not shown. |
| `schema.py` | Output fields (`BriefText`, passed to Ollama as a JSON schema) and the Markdown renderer. |
| `validate.py` | Checks number grounding, required facts, banned wording, JSON or field-name leaks, and US spellings. |
| `template_brief.py` | Deterministic brief. It is the fallback and the evaluation baseline. |
| `brief.py` | Orchestration: prompt, then Ollama, then validation, retry and fallback, then trace. |
| `llm_client.py` | Ollama wrapper. Records the model digest and server version. |
| `eval/` | Fixed 20-road test sample, batch runner, and human rating rubric. |

## Prompt versions

| Version | Files | What it is |
|---|---|---|
| `brief_v1` | `brief_v1.j2`, `brief_v1_example.json` | One worked example (flagged road, PCI and rutting) |
| `brief_v2` (**default**) | `brief_v2.j2`, `brief_v2_examples.json` | Adds a `numbers_to_include` checklist to each section (the same list `validate.completeness()` checks), a fixed sentence pattern for the repair field, and a second worked example (not flagged, PCI only) |

Choose a version with `--prompt brief_v1` on `llm.cli` or `run_eval`, or set `$env:LLM_PROMPT`. Old versions are never edited. Any change goes into a new `brief_vN` entry in the `PROMPTS` dictionary in `brief.py`. Each trace records the prompt name and a hash of its files. Worked examples must come from training-split roads and must pass `validate()`; the tests check both.

`python -m llm.eval.run_eval compare` compares the latest run of each prompt and engine pair. It pairs each road's first-try result across versions and applies an exact McNemar test to those pairs.

## Things to know

- **Model loading workaround.** The shipped `V0.0.4/models/*.joblib` files were pickled on macOS, and their learner *config* block will not load under Windows xgboost 3.4.1. `v002.booster()` loads only the *model* section: the trees, base score, objective and feature names. Nothing is refitted. The tests confirm the loaded models reproduce `results/` exactly: PCI test RMSE 14.160, rutting 1.627, and classifier TN 134 / FP 9 / FN 8 / TP 31.
- **Action tiers are provisional.** The rules in `payload._action()` still need confirming with Luke and the client:
  - **corrective:** the repair classifier flags the road, predicted PCI < 55, or predicted rutting is high severity
  - **preventive:** predicted PCI < 70, or predicted rutting is medium severity
  - **investigate data:** the PCI calculation reported an error
  - **monitor:** none of the above
- **Held back from the brief:** there is no claim that the PCI model beats the age-trend baseline until CLAUDE.md §6.1 (0.802 vs 0.708) is settled. The rutting brief always states that the model is not distinguishable from persistence, because the bootstrap result in `results/` says so.
- **Demo-speed numbers:** all error figures are read from `V0.0.4/results/` at run time, so they update automatically after the full `PCI_SEARCH_ITER=40` run.
- **Pins:** pandas 2.3.3 and numpy 2.2.6 match `requirements-lock.txt`. xgboost 3.4.1, shap 0.52.0, scikit-learn 1.9.1 and joblib 1.6.0 are what this environment used, and are offered as pins for the lock file (Decision Log #18).
