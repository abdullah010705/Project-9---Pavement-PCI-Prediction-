# LLM Plan — Pavement Deterioration Project (ZEIT3101, Group 9)

> **Integration note (added when the LLM layer was merged into the pipeline).**
> `llm/` now lives *inside* the pipeline folder rather than beside a separate `V0.0.3/`.
> `v002.py` detects either layout by looking for `config.py`, and `V002_DIR` still
> overrides it, so nothing else changed. `run_llm.sh` is the macOS/Linux equivalent of
> `run_llm.cmd` / `run_llm.ps1`, and `./run.sh --with-llm` runs the briefs as an optional
> pipeline stage. See the project README for details.



**Owner:** Luke
**Started:** 30 Sep 2026
**Status:** Draft v0.4. Component E v1 is built. Prompt `brief_v2` has been evaluated (§10a). The scope decisions in §9 are still open.
**Builds on:** `V0.0.3/` (read-only), as described in `CLAUDE.md`

---

## 0. Project brief (source of scope)

### 0.1 Outline as issued

> **ZEIT3101 IT Project 2**
> **Can Artificial Intelligence Predict Road Failures? A Data-Driven Study of Pavement Deterioration**
>
> **Project description**
> - Road pavement deterioration is a significant challenge for transport agencies, as timely maintenance decisions are essential for ensuring road safety and minimising lifecycle costs. Traditional pavement management systems often rely on empirical deterioration models that may not fully capture the complex interactions among traffic loading, environmental conditions, pavement characteristics, and maintenance history.
> - This project aims to investigate the application of machine learning techniques to predict pavement deterioration using historical pavement condition data. The student will review existing deterioration modelling approaches, develop and compare several machine learning models, and evaluate their predictive performance using real-world pavement datasets. Potential input variables may include pavement age, traffic volume, climatic conditions, structural characteristics, and historical condition ratings.
> - The project will provide the student with experience in data analytics, machine learning, and infrastructure asset management. Depending on data availability and student interest, the project may also explore the use of Large Language Models (LLMs) for interpreting maintenance records and generating explainable insights from prediction results.
>
> **Objectives**
> 1. Review current pavement deterioration modelling approaches.
> 2. Preprocess pavement condition datasets.
> 3. Develop and compare machine learning models for deterioration prediction.
> 4. Investigate the potential role of AI tools, including LLMs, in supporting pavement asset management decision-making.
>
> **Expected skills:** basic programming (Python preferred); interest in data analytics, AI and transport infrastructure; no prior ML experience required.
>
> **Expected deliverables**
> - A curated and pre-processed pavement condition dataset suitable for model development.
> - Machine learning models for pavement deterioration prediction.
> - A final thesis report documenting methodology, results, limitations, and recommendations for future work.
> - An oral presentation and demonstration of the developed prediction framework.
>
> **Required infrastructure:** a standard desktop or laptop computer capable of running Python-based data analysis and machine learning software.

### 0.2 What the brief means for this workstream

| Brief says | Implication for the LLM plan |
|---|---|
| The LLM work is something the project "*may also explore*", "*depending on data availability and student interest*" | The LLM work is an **extension**, not a core deliverable. It must not take time from Objectives 1–3 or from Luke's open pipeline items (CLAUDE.md §6). The "data availability" condition is exactly the Component M problem (§6.1), so a documented, evidence-based scope decision is a legitimate outcome in itself. |
| "*Interpreting maintenance records*" | Component M (§6) |
| "*Generating explainable insights from prediction results*" | Component E (§5). It sits on top of the Objective 3 models and SHAP outputs. |
| Objective 4: "*investigate the potential role*" of AI tools "*in supporting … decision-making*" | The thesis contribution is an **evaluation** of whether LLM output helps an asset manager and can be trusted (§7), not just a working tool. Negative or mixed findings are still valid results. |
| Traditional empirical models may miss interactions involving "*maintenance history*" | Motivates Component M: turning maintenance history into usable context is a gap the brief itself points to. |
| Deliverable: a thesis covering "*limitations, and recommendations for future work*" | Record LLM limitations as we go (small local models, synthetic data if used, demo-run figures, SHAP ≠ causation). Record future work too, e.g. real agency work-order text or larger models. |
| Deliverable: "*demonstration of the developed prediction framework*" | Component E is the natural demo layer: prediction → explained brief for one road, run live (Milestone 6). |
| Infrastructure: a "*standard desktop or laptop*" | Local Ollama fits the brief: no cloud cost and no data leaves the machine. The RTX 4070 Ti SUPER is above "standard". Test the demo on a CPU-only laptop with a small model (e.g. `phi4-mini`), or pre-generate briefs as a fallback. |
| Skills: "*Python preferred*" | Python throughout. Keep the Ollama layer thin so the team can read and maintain it. |

---

## 1. Purpose

Objective 4 of the project outline asks us to *investigate the role of AI tools, including LLMs, in supporting pavement asset-management decisions*. The outline names two uses, and this plan covers both:

| Use | Outline wording | Component |
|---|---|---|
| (a) | Interpreting maintenance records | **M: Maintenance interpreter** |
| (b) | Generating explainable insights from prediction results | **E: Explainable condition brief** |

Both components run on a **local LLM served by Ollama**. All other code is **Python**. The LLM only writes language. Every number comes from `V0.0.3` results, models or data through Python code.

---

## 2. Design principles (non-negotiable)

1. **The LLM never generates a number.** Python code builds a structured *fact payload*, and the LLM may only use figures that are in it. A validator extracts every number from the output and rejects the output if a number is not in the payload (allowing for rounding and unit conversion).
2. **Read-only against `V0.0.3`.**
   - Import `config`, `schema_checks` and `results_io`.
   - Load `models/*.joblib`.
   - Never retrain and never edit team-owned files.
   - Never touch `data/` or the split.
3. **Traceability.** Every generated output is stored with:
   - its input payload
   - the prompt template name and version
   - the Ollama model tag and digest
   - the generation options (temperature, seed)
   - a timestamp
   - the validator result
4. **Honest uncertainty.** Each brief includes the model's test or CV error. Rutting predictions must not be described as better than persistence, because the bootstrap result says they cannot be told apart.
5. **SHAP describes the model, not the road.** Write "the model's estimate was pulled down mainly by…", never "the road will fail because…".
6. **Units.**
   - PCI stays on its 0–100 scale.
   - Rut depth is converted to mm (× 3.175) in the payload, not by the LLM.
   - IRI and climate units must be verified before they are labelled.
7. **Australian English** in prompts, outputs and docs.
8. **Reproducibility.**
   - Set temperature to 0 and use a fixed `seed`.
   - Pin the model tag.
   - Record the Ollama version (currently 0.24.0).

---

## 3. Environment (checked 30 Sep 2026)

| Item | State |
|---|---|
| GPU | NVIDIA RTX 4070 Ti SUPER, 16 GB VRAM. This runs 7–8B models at full speed, and quantised ~14B models fit. |
| Ollama | 0.24.0, installed |
| Local models already pulled | `llama3.1:8b`, `qwen3:8b`, `mistral:7b`, `phi4-mini`, `nomic-embed-text` |
| Python | 3.12.10 available (`py -3.12`). Use 3.12 to match the team's environment. |
| `V0.0.3/.venv` | macOS venv. Ignore it. **Built instead:** `%USERPROFILE%\.venvs\zeit3101` (Python 3.12.10), kept outside OneDrive so package files don't sync. Pins are in `llm/requirements.txt`. |
| Shipped `models/*.joblib` | Won't load with `joblib.load` on Windows: xgboost rejects the serialised learner config. `llm/v002.py` loads only the model section instead. **No refit.** Tests confirm the models exactly reproduce the `results/` metrics. |

Python dependencies for the LLM layer, on top of `V0.0.3/requirements.txt`:
- `ollama`: the official Python client
- `pydantic`: payload and output schemas
- `jinja2`: prompt templates
- `pytest`
- optionally `streamlit` for the demo

---

## 4. Architecture

```
            V0.0.3 (read-only)
  data/ · models/*.joblib · results/*.csv|json · config.py
                     │
                     ▼
   ┌──────────────────────────────────┐
   │ 1. Fact layer (Python)            │  payload.py
   │   predictions, SHAP top-k,        │  → pydantic Payload (JSON)
   │   repair probability, error       │
   │   context, unit conversions,      │
   │   maintenance events              │
   └──────────────────────────────────┘
                     │
                     ▼
   ┌──────────────────────────────────┐
   │ 2. Prompt layer                   │  prompts/*.j2 (versioned)
   │   system rules + payload JSON     │
   └──────────────────────────────────┘
                     │
                     ▼
   ┌──────────────────────────────────┐
   │ 3. Ollama (local)                 │  ollama.chat(..., format=schema,
   │   structured JSON output          │   options={temperature:0, seed})
   └──────────────────────────────────┘
                     │
                     ▼
   ┌──────────────────────────────────┐
   │ 4. Validator                      │  validate.py
   │   numbers ⊆ payload, required     │  on failure: retry once, else
   │   fields, banned phrasing         │  fall back to a template brief
   └──────────────────────────────────┘
                     │
                     ▼
   ┌──────────────────────────────────┐
   │ 5. Store + render                 │  outputs/*.jsonl, HTML/Markdown
   └──────────────────────────────────┘
```

We use Ollama's **structured outputs** (the `format` parameter takes a JSON schema) so that responses are machine-checkable fields, not free prose.

The code also always produces a deterministic, non-LLM **template brief** from the same payload. It is the fallback when validation fails, and it is the baseline the LLM output is judged against in §7.

---

## 5. Component E: Explainable condition brief (outline use (b))

### 5.1 Input: fact payload for one `ROAD_ID` + `SURVEY_DATE`

The payload has five parts:
- **Identity:** `PHYSICAL_ROAD_ID`, `ROAD_ID`, survey date, split (train/test).
- **Current condition:** `PCI_SCORE`, `PCI_RATING`, `PCI_DOMINANT_DISTRESS`, rut depth (mm), IRI, age, cumulative ESAL.
- **Predictions:**
  - `PCI_NEXT` from `xgboost_pci.joblib`
  - `RUT_NEXT` from `xgboost_rut.joblib`, converted to mm
  - repair probability from `repair_classifier.joblib`, flagged at `config.REPAIR_THRESHOLD` (PCI < 55)
- **Drivers:** the top 3–5 SHAP contributions for each target, with sign and magnitude and human-readable feature labels from a lookup table in `llm/`.
- **Error context:** test and grouped-CV RMSE/MAE read from `results/`, the bootstrap verdict against the baselines, and the classifier's precision and recall.

Where possible, reuse the existing files: `predictions_*.csv`, `shap_values_*.csv` and `individual_explanations_*.csv`. Only score rows live when a row is not already covered, and use the feature order from `reduced_features_*.json`.

### 5.2 Output schema (the LLM fills in these fields)

- `headline`: one sentence.
- `condition_now`: the current state.
- `outlook`: the predicted next condition, with its uncertainty band.
- `drivers`: 3–4 bullets, phrased at model level.
- `repair_flag`: the flag with its caveat.
- `recommended_action_tier`: chosen from a **fixed list** that is defined in code, not by the LLM:
  - monitor
  - preventive
  - corrective
  - investigate data
- `caveats`

### 5.3 Data-quality guard

Some rows have implausible actual values, for example actual PCI 0.0 while current PCI is high (see CLAUDE.md §7). The payload builder flags these by checking `PCI_ERROR` in the raw file and looking for suspected unrecorded rebuilds. The brief must then say "possible data artefact" and must not narrate a failure.

---

## 6. Component M: Maintenance record interpreter (outline use (a))

### 6.1 The data problem (decide before building)

`V0.0.3` has **no free-text maintenance records**. Maintenance appears only as:
- `PRIOR_PATCH_COUNT`
- `DAYS_SINCE_LAST_PATCH` (about 93% missing)
- `LAST_PATCH_DATE` (raw file only)
- per-survey patching and pothole distress quantities

There is nothing for an LLM to "interpret" until we pick one of these sources:

| Option | Source | Pros | Cons |
|---|---|---|---|
| **M1: LTPP coded records** | LTPP InfoPave maintenance (MNT) and rehabilitation (RHB) tables for the same GPS-1/GPS-2 sections | Real and public. Joins on `STATE_CODE` + `SHRP_ID`. The LLM turns coded work types into plain-English histories and links them to condition jumps. | New data pull. The records are coded, not narrative. Joining must not change `data/` or the split. |
| **M2: Synthetic narratives** | Records generated from the existing patch and distress columns, clearly labelled as synthetic | Quick. Shows extraction (text → structured fields) with a known ground truth, which makes accuracy measurable. | Not real evidence. Must be declared a limitation in the thesis. |
| **M3: Hybrid** | M1 codes rendered as realistic work-order text (M2 style), then the LLM extracts them back into structured fields | Real events plus a measurable extraction task | The most work |

**Recommended: M3 if time allows, otherwise M1.** M3 gives two thesis results:
1. **Summarisation:** a road's maintenance history in plain English.
2. **Extraction accuracy:** work type, date, quantity and location recovered from text, scored against the known codes.

This needs sign-off from the supervisor or client (see §9).

### 6.2 Pipeline (for M1/M3)

1. `maintenance_ingest.py`
   - Load the LTPP MNT/RHB exports into `llm/data_external/`, never into `V0.0.3/data/`.
   - Map work-type codes to descriptions using a lookup table in code.
2. `maintenance_payload.py`
   - Build a per-road event timeline.
   - Align it with the survey dates.
   - Compute the condition change around each event. This is done in Python.
3. The LLM task is one of:
   - **(i)** summarise the timeline, or
   - **(ii)** for M3, extract structured fields from the work-order text, with Ollama constrained to a JSON schema.
4. The validator checks that dates, quantities and work types match the source.
5. Optional: add the maintenance summary to the Component E brief as context. It is never used as a model input, because that would change the models.

---

## 7. Evaluation (a thesis result, not only a demo)

**Fixed evaluation sample:** 20 **test-split** roads, stratified by PCI band (Good / Fair / Poor) × repair flag. The sample is chosen once, with a fixed seed, and saved to `llm/eval/sample.csv`.

| Measure | Method | Component |
|---|---|---|
| Factual consistency | Automated: the share of numbers in the output that match the payload. Target: 100% after validation. We report the pre-validation rate as the finding. | E, M |
| Schema validity | The share of responses that parse on the first try | E, M |
| Forbidden phrasing | Regex for causal SHAP wording ("will fail because"), rutting over-claims and US units | E |
| Extraction accuracy | Field-level precision, recall and F1 against the known codes | M (M2/M3) |
| Human rating | Rubric scored 1–5 for accuracy, usefulness and clarity. At least two raters (Luke plus one team member). Report inter-rater agreement. | E, M |
| Model comparison | Run the same sample on 2–3 Ollama models, e.g. `llama3.1:8b`, `qwen3:8b` and `mistral:7b` | E, M |
| Baseline | The deterministic template brief. Does the LLM add readability without losing accuracy? | E |
| Latency | Seconds per brief on the RTX 4070 Ti SUPER | E, M |

---

## 8. Folder layout (new, sibling to `V0.0.3`)

```
AAA. LLM/
  CLAUDE.md
  llm-plan.md              ← this file
  V0.0.3/                  ← team code, read-only
  llm/
    __init__.py
    v002.py                # the only place that adds V0.0.3 to sys.path; imports config etc.
    payload.py             # Component E fact payload
    features_lookup.py     # feature name → plain label + unit
    maintenance_ingest.py  # Component M
    maintenance_payload.py
    llm_client.py          # thin Ollama wrapper (model, seed, temperature, format)
    validate.py            # number grounding, schema, phrasing checks
    template_brief.py      # deterministic non-LLM baseline
    store.py               # JSONL trace writer
    cli.py                 # `python -m llm.cli brief --road 48-1077-1 --date 2004-06-01`
    prompts/
      brief_v1.j2
      maint_summary_v1.j2
      maint_extract_v1.j2
    eval/
      sample.csv
      run_eval.py
      rubric.md
    data_external/         # LTPP MNT/RHB pulls (Component M)
    outputs/               # generated briefs + traces (git-ignored)
  tests/
    test_payload.py
    test_validate.py
```

---

## 9. Open decisions

| # | Question | Blocks |
|---|---|---|
| 1 | Component M data source: M1, M2 or M3? Supervisor/client sign-off needed if synthetic data is used. | Milestone 4 |
| 2 | Primary Ollama model: start with `llama3.1:8b` or `qwen3:8b` and choose on the eval? Do we try a ~14B model? | Milestone 3 |
| 3 | Demo format: CLI, Streamlit, or a static HTML report per road | Milestone 5 |
| 4 | Is this Luke's individual allocation or a team deliverable? Who else uses it? | Scope |
| 5 | Dates for Review 2 and the final demo | Timeline |
| 6 | Which figures go into the error context while `results/` is still a demo-speed run? Proposal: build against the current files and regenerate after the full `PCI_SEARCH_ITER=40` run. | Final numbers |
| 7 | The anchored baseline (0.802 vs 0.708, CLAUDE.md §6.1) changes what the brief can say about ML versus the baseline. Keep that claim out of briefs until it is settled. | Brief wording |
| 8 | Action-tier rules in `llm/payload.py` are provisional: corrective = flagged, or PCI < 55, or high-severity rutting; preventive = PCI < 70 or medium-severity rutting. Confirm them with the client. | Brief wording |
| 9 | May the pipeline be re-run on Luke's machine? It overwrites the shared `results/` and `models/`. The alternative is to run it on a copy. | Milestone 0 |

---

## 10. Milestones

| # | Milestone | Done when | Status (30 Sep 2026) |
|---|---|---|---|
| 0 | Environment | The venv works. The `V0.0.3` pipeline runs at `PCI_SEARCH_ITER=10`. `ollama` client reaches the local server. | **Partly done.** The venv and Ollama work. The pipeline has **not** been re-run: a run overwrites the shared `V0.0.3/results/` and `models/`, which needs Luke's go-ahead. |
| 1 | Model access | All three joblib models load and score one row with the correct reduced feature order. | **Done**, using the loader workaround (§3). The models reproduce `results/` exactly. |
| 2 | Fact payload | `payload.py` returns a validated payload for any test row. Unit tests pass. The template brief renders. | **Done.** 17 tests pass, and the template brief is valid for all 183 test surveys. |
| 3 | Component E v1 | Prompt, Ollama call and validator work end to end, and trace records are written. | **Done.** Prompt `brief_v1` has one worked example. The flow is validate, retry once, then fall back to the template. |
| 4 | Component M v1 | Data source decided and ingested. Summary and/or extraction work with validation. | Blocked on §9 #1 |
| 5 | Evaluation | Automated metrics run on the 20-road sample for 2–3 models. Human rubric completed. | Two prompt versions evaluated on the same 20 roads (see §10a). Human rating not yet done. |
| 6 | Demo + write-up | Demo front end done. Thesis section written: method, results, limitations (local 8B models, synthetic data if used, demo-run figures, SHAP ≠ causation). | Not started |

### 10a. Evaluation log

Same fixed 20-road test sample, temperature 0, seed 42, and one retry allowed. The "first-try pass" and "accepted but wrong" columns use the **current** validator, including the meaning checks added on 30 Sep. Run `python -m llm.eval.run_eval rescore` to reproduce them.

| Prompt | Model | First-try pass | Accepted after ≤1 retry | Accepted but wrong (driver direction flipped) | Median s/brief |
|---|---|---|---|---|---|
| `brief_v1` | qwen3:8b | 45% | 100% | 0 | 8.7 |
| `brief_v1` | llama3.1:8b | 40% | 100% | 1 | 20.8 |
| `brief_v2` | qwen3:8b | **100%** | 100% | 0 | 6.7 |
| `brief_v2` | llama3.1:8b | 90% | 100% | 2 | 14.2 |

Paired first-try results, v1 → v2, using the validator in force at run time. llama3.1: 12 roads improved and 0 got worse (exact McNemar p = 0.0005). qwen3: 11 improved and 0 got worse (p = 0.001).

**Findings:**
1. Almost all v1 failures were omissions in the repair field. A per-field checklist (`numbers_to_include`) and a fixed sentence pattern removed them.
2. With v2, no model invented a number on the first try, down from 2 briefs with v1.
3. Checking numbers alone is not enough. llama3.1 reversed a SHAP driver's direction in 3 of 40 accepted briefs across both runs, because numbers are matched ignoring sign. The meaning checks in `validate.py` (driver direction and repair-flag wording) now reject these. qwen3 made no such errors.
4. **qwen3:8b is recommended** as the main model: more reliable, and about twice as fast.

---

## 11. Risks

- **Small local models hallucinating numbers or causes.** Mitigated by structured output, the validator, retry-then-template fallback, and temperature 0.
- **`results/` figures changing after the full run.** Everything is read from files at run time, so nothing is hardcoded.
- **Test-set leakage.** The LLM layer never trains or tunes anything, and the evaluation sample comes only from the test split.
- **Scope creep in Component M.** Time-box the LTPP data pull. Fall back to M2 with an explicit limitation.

---

## 12. Change log

| Date | Version | Change |
|---|---|---|
| 30 Sep 2026 | 0.1 | Initial plan |
| 30 Sep 2026 | 0.2 | Added §0: project brief and what it means for this workstream |
| 30 Sep 2026 | 0.3 | Component E v1 built (`llm/`). Environment and model-loading notes added. Milestone status column. Decisions #8–9 added. The §8 layout differs slightly as built; `llm/README.md` is the current map. |
| 30 Sep 2026 | 0.4 | Added `run_llm.ps1`. Prompt `brief_v2` is now the default and has been evaluated (§10a). Meaning checks added to the validator. |
