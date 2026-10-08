"""
run_eval.py — automated evaluation of condition briefs (llm-plan.md §7).

    python -m llm.eval.run_eval sample                   # build eval/sample.csv (once)
    python -m llm.eval.run_eval run --models qwen3:8b llama3.1:8b [--prompt brief_v2]
    python -m llm.eval.run_eval compare               # paired comparison of all runs
    python -m llm.eval.run_eval rescore               # re-check saved briefs, current validator

The sample is TEST-split surveys only, at most one per physical road, stratified by
current PCI band × repair flag, drawn with a fixed seed. It is written once and
reused so every model and prompt version is scored on the same roads.

Outputs (llm/eval/results/):
  briefs_<run>.jsonl   full trace per brief (payload, prompt hash, model digest, attempts)
  summary_<run>.csv    one row per brief: source, attempts, first-try errors, latency
  rating_<run>.csv     blank human-rating sheet (see rubric.md)
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

from llm import v002  # noqa: E402
from llm import brief as brief_mod  # noqa: E402
from llm.brief import generate  # noqa: E402
from llm.payload import build_payload, list_surveys  # noqa: E402

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(EVAL_DIR, "sample.csv")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
SAMPLE_SIZE = 20
SEED = v002.config.RANDOM_SEED


def _band(pci: float) -> str:
    return "good (85+)" if pci >= 85 else "fair (55-85)" if pci >= 55 else "poor (<55)"


def make_sample(n: int = SAMPLE_SIZE, seed: int = SEED) -> pd.DataFrame:
    if os.path.exists(SAMPLE_PATH):
        raise FileExistsError(f"{SAMPLE_PATH} exists; the sample is fixed. Delete it "
                              "deliberately if it must be redrawn.")
    s = list_surveys("test")
    s = s[s.targets.str.contains("pci")].copy()  # repair flag needs the PCI row
    s["physical_road_id"] = s.ROAD_ID.str.rsplit("-", n=1).str[0]
    s["pci_band"] = s.pci.map(_band)
    s["repair_flag"] = [build_payload(r.ROAD_ID, r.SURVEY_DATE_ISO).repair.flagged
                        for r in s.itertuples()]
    rng = np.random.default_rng(seed)
    s = s.iloc[rng.permutation(len(s))]
    s = s.drop_duplicates("physical_road_id")  # one survey per road

    cells = list(s.groupby(["pci_band", "repair_flag"]))
    quota = {k: 0 for k, _ in cells}
    while sum(quota.values()) < n:  # round-robin so small cells are represented
        added = False
        for k, g in cells:
            if quota[k] < len(g) and sum(quota.values()) < n:
                quota[k] += 1
                added = True
        if not added:
            break
    out = pd.concat([g.head(quota[k]) for k, g in cells])
    out = out[["ROAD_ID", "SURVEY_DATE_ISO", "physical_road_id", "pci", "pci_band",
               "repair_flag", "targets"]].reset_index(drop=True)
    out.to_csv(SAMPLE_PATH, index=False)
    print(out.groupby(["pci_band", "repair_flag"]).size())
    return out


def run(models: list[str], include_template: bool = True,
        prompt: str | None = None) -> pd.DataFrame:
    prompt = prompt or brief_mod.PROMPT_NAME
    sample = pd.read_csv(SAMPLE_PATH)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = []
    trace_path = os.path.join(RESULTS_DIR, f"briefs_{run_id}.jsonl")
    engines = ([("template", None)] if include_template else []) + [("llm", m) for m in models]
    with open(trace_path, "w", encoding="utf-8") as trace:
        for r in sample.itertuples():
            p = build_payload(r.ROAD_ID, r.SURVEY_DATE_ISO)
            for kind, model in engines:
                rec = generate(p, model=model or "", use_llm=kind == "llm", store=False,
                               prompt=prompt)
                rec["eval_engine"] = model or "template"
                trace.write(json.dumps(rec, default=str) + "\n")
                first = rec["attempts"][0]["validation"] if rec["attempts"] else rec["validation"]
                rows.append({
                    "road_id": r.ROAD_ID, "survey_date": r.SURVEY_DATE_ISO,
                    "pci_band": r.pci_band, "repair_flag": r.repair_flag,
                    "engine": model or "template",
                    "prompt": prompt if kind == "llm" else "",
                    "source": rec["source"],
                    "attempts": len(rec["attempts"]),
                    "first_try_ok": bool(first["ok"]),
                    "first_try_errors": " | ".join(first.get("errors", [])),
                    "first_try_unsupported_numbers": " ".join(
                        first.get("numbers_unsupported", []) or []),
                    "final_ok": rec["validation"]["ok"],
                    "numbers_checked": rec["validation"]["numbers_checked"],
                    "seconds": sum(a["seconds"] for a in rec["attempts"]),
                    "warnings": " | ".join(rec["validation"]["warnings"]),
                })
                print(f"{r.ROAD_ID:12s} {model or 'template':14s} {rec['source']:18s} "
                      f"first_ok={first['ok']}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, f"summary_{run_id}.csv"), index=False)

    rating = df[["road_id", "survey_date", "engine", "source"]].copy()
    for c in ("accuracy_1to5", "usefulness_1to5", "clarity_1to5", "rater", "comment"):
        rating[c] = ""
    # Shuffle so raters do not see engines grouped together
    rating = rating.sample(frac=1, random_state=SEED)
    rating.to_csv(os.path.join(RESULTS_DIR, f"rating_{run_id}.csv"), index=False)

    print("\n" + summarise(df).to_string())
    print(f"\ntrace: {trace_path}")
    return df


def summarise(df: pd.DataFrame, by="engine") -> pd.DataFrame:
    df = df.copy()
    # Empty strings come back from CSV as NaN
    df["first_try_unsupported_numbers"] = (df.first_try_unsupported_numbers
                                           .fillna("").astype(str))
    g = df.groupby(by)
    return pd.DataFrame({
        "briefs": g.size(),
        "first_try_pass_%": (100 * g.first_try_ok.mean()).round(0),
        "llm_accepted_%": (100 * g.source.apply(
            lambda s: s.isin(["llm", "llm_retry"]).mean())).round(0),
        "fell_back_to_template": g.source.apply(lambda s: (s == "template_fallback").sum()),
        "first_try_invented_numbers": g.first_try_unsupported_numbers.apply(
            lambda s: (s.str.len() > 0).sum()),
        "final_valid_%": (100 * g.final_ok.mean()).round(0),
        "median_seconds": g.seconds.median().round(1),
    })


def load_runs() -> pd.DataFrame:
    """Every summary_*.csv, tagged with its run id. Runs before the prompt column
    existed used brief_v1."""
    import glob
    frames = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "summary_*.csv"))):
        d = pd.read_csv(path)
        d["run"] = os.path.basename(path)[len("summary_"):-len(".csv")]
        if "prompt" not in d:
            d["prompt"] = np.where(d.engine == "template", "", "brief_v1")
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def compare() -> pd.DataFrame:
    """Per (prompt, engine) summary of the latest run of each, plus paired first-try
    outcomes on the same roads between prompt versions."""
    d = load_runs()
    d = d[d.engine != "template"]
    latest = d.groupby(["prompt", "engine"]).run.transform("max")
    d = d[d.run == latest]
    table = summarise(d, by=["prompt", "engine"])
    print(table.to_string())

    prompts = sorted(d.prompt.unique())
    for engine in sorted(d.engine.unique()):
        for a_, b_ in zip(prompts, prompts[1:]):
            A = d[(d.engine == engine) & (d.prompt == a_)].set_index("road_id").first_try_ok
            B = d[(d.engine == engine) & (d.prompt == b_)].set_index("road_id").first_try_ok
            j = pd.concat([A, B], axis=1, keys=[a_, b_]).dropna()
            if j.empty:
                continue
            only_b = int((~j[a_] & j[b_]).sum())
            only_a = int((j[a_] & ~j[b_]).sum())
            print(f"\n{engine}: first-try pass, {a_} -> {b_} on {len(j)} paired roads: "
                  f"both {int((j[a_] & j[b_]).sum())}, only {b_} {only_b}, "
                  f"only {a_} {only_a}, neither {int((~j[a_] & ~j[b_]).sum())}; "
                  f"exact McNemar p = {_mcnemar(only_a, only_b):.3g}")
    return table


def rescore() -> pd.DataFrame:
    """Re-validate every saved LLM attempt with the CURRENT validator (no new LLM calls).

    Shows what a stricter validator would have caught in earlier runs. The accepted
    brief of each run is also re-checked: "accepted_now_invalid" counts briefs that
    were shown to the user under the old validator but would now be rejected.
    """
    import glob
    from llm.payload import Payload
    from llm.schema import BriefText
    from llm.validate import validate
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "briefs_*.jsonl"))):
        run_id = os.path.basename(path)[len("briefs_"):-len(".jsonl")]
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["eval_engine"] == "template":
                    continue
                p = Payload(**rec["payload"])

                def check(raw):
                    try:
                        return validate(BriefText.model_validate_json(raw), p)
                    except Exception:  # noqa: BLE001
                        return None
                first = check(rec["attempts"][0]["raw"])
                final = validate(BriefText(**rec["brief"]), p)
                rows.append({"run": run_id, "prompt": (rec["prompt"] or {}).get("name"),
                             "engine": rec["eval_engine"], "road_id": rec["road_id"],
                             "first_try_ok_now": bool(first and first.ok),
                             "accepted_now_invalid": rec["source"] != "template_fallback"
                             and not final.ok,
                             "new_errors": " | ".join(final.errors)})
    d = pd.DataFrame(rows)
    g = d.groupby(["prompt", "engine"])
    table = pd.DataFrame({"briefs": g.size(),
                          "first_try_pass_%_now": (100 * g.first_try_ok_now.mean()).round(0),
                          "accepted_now_invalid": g.accepted_now_invalid.sum()})
    print(table.to_string())
    bad = d[d.accepted_now_invalid]
    if len(bad):
        print("\nAccepted briefs the current validator rejects:")
        print(bad[["prompt", "engine", "road_id", "new_errors"]].to_string(index=False))
    return d


def _mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="llm.eval.run_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample")
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+", default=["qwen3:8b", "llama3.1:8b"])
    r.add_argument("--no-template", action="store_true")
    r.add_argument("--prompt", choices=sorted(brief_mod.PROMPTS),
                   default=brief_mod.PROMPT_NAME)
    sub.add_parser("compare")
    sub.add_parser("rescore")
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        make_sample()
    elif a.cmd == "compare":
        compare()
    elif a.cmd == "rescore":
        rescore()
    else:
        run(a.models, include_template=not a.no_template, prompt=a.prompt)


if __name__ == "__main__":
    main()
