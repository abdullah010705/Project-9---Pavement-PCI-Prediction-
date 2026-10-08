"""
llm_stage.py — pipeline wrapper for the LLM condition briefs (Component E, owner: Luke).

Thin adapter only. All the work is in the `llm/` package; this just picks a few
representative roads and writes their briefs into results/ so the stage produces
something inspectable alongside every other stage's output.

Optional by design — run with `--with-llm`. A real brief needs a local Ollama model;
without one the layer falls back to `llm/template_brief.py`, which is deterministic and
produces the same facts without the generated prose. Either way the numbers are built by
code and checked by `llm/validate.py`, never written by the model.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

import config

N_SAMPLE = 3


def _candidates():
    """A few test-split surveys worth showing: one flagged for repair, one not, and one
    that has both PCI and rutting predictions."""
    from llm import v002

    df = v002.dataset("pci")
    test = df[df[config.SPLIT_COL] == config.TEST_VALUE].copy()
    if "NEEDS_REPAIR" in test.columns:
        flagged = test[test.NEEDS_REPAIR == 1]
        clear = test[test.NEEDS_REPAIR == 0]
    else:
        flagged, clear = test.iloc[:0], test
    picks = []
    for frame in (flagged, clear):
        if len(frame):
            r = frame.iloc[0]
            picks.append((str(r[config.ROAD_ID_COL]), str(r[config.SURVEY_DATE_COL])[:10]))
    rut_ids = set(v002.dataset("rut")[config.ROAD_ID_COL].astype(str))
    both = test[test[config.ROAD_ID_COL].astype(str).isin(rut_ids)]
    if len(both):
        r = both.iloc[0]
        picks.append((str(r[config.ROAD_ID_COL]), str(r[config.SURVEY_DATE_COL])[:10]))
    seen, out = set(), []
    for p in picks:
        if p not in seen:
            seen.add(p); out.append(p)
    return out[:N_SAMPLE]


def run():
    if config.PROJECT_ROOT not in sys.path:
        sys.path.insert(0, config.PROJECT_ROOT)

    try:
        from llm.brief import brief_for
    except ImportError as exc:
        raise NotImplementedError(
            f"LLM layer not available ({exc}). Install its dependencies with "
            f"'pip install -r llm/requirements.txt', or run without --with-llm."
        ) from exc

    use_llm = True
    try:
        import ollama
        ollama.list()
    except Exception:
        use_llm = False
        print("[llm] Ollama not reachable — using the deterministic template fallback. "
              "Facts and validation are unchanged; only the prose differs.")

    rows = []
    for road, date in _candidates():
        try:
            record = brief_for(road, date, use_llm=use_llm)
        except Exception as exc:
            print(f"[llm] {road} {date}: skipped ({type(exc).__name__}: {exc})")
            continue

        validation = record.get("validation") or {}
        rows.append({
            "road_id": road, "survey_date": date,
            "source": record.get("source"),
            "validation_ok": validation.get("ok"),
            "numbers_checked": validation.get("numbers_checked"),
            "attempts": len(record.get("attempts") or []),
            "markdown": record.get("markdown") or "",
        })
        print(f"[llm] brief generated: {road} {date} "
              f"(source={rows[-1]['source']}, "
              f"{rows[-1]['numbers_checked']} numbers checked, "
              f"valid={rows[-1]['validation_ok']})")

    if not rows:
        print("[llm] no briefs generated")
        return []

    sources = {r["source"] for r in rows}
    if sources == {"template"}:
        provenance = ("> **These briefs were written by the deterministic template, not by a "
                      "language model.** Ollama was not reachable when this ran, so the layer "
                      "used its fallback path (`llm/template_brief.py`). The facts and the "
                      "validation are identical; only the wording differs. Do not present "
                      "these as LLM output — install Ollama and pull a model, then re-run.")
    elif "template" in sources:
        provenance = ("> **Mixed sources.** Some briefs below came from the language model and "
                      "some from the deterministic template fallback. The per-brief source is "
                      "recorded in `llm_brief_index.csv`.")
    else:
        provenance = ("> Written by a local language model via Ollama, then validated: every "
                      "number in the text was checked against the facts the pipeline computed.")

    out_md = os.path.join(config.RESULTS_DIR, "llm_sample_briefs.md")
    with open(out_md, "w") as f:
        f.write("# Sample condition briefs\n\n"
                "Produced by the LLM layer (`llm/`, Component E). Every number is built in "
                "code and checked against the model outputs before a brief is accepted.\n\n"
                + provenance + "\n\n---\n\n")
        for r in rows:
            f.write(f"*(source: {r['source']}, {r['numbers_checked']} numbers checked, "
                    f"validation {'passed' if r['validation_ok'] else 'FAILED'})*\n\n")
            f.write(r["markdown"].rstrip() + "\n\n---\n\n")

    pd.DataFrame([{k: v for k, v in r.items() if k != "markdown"} for r in rows]).to_csv(
        os.path.join(config.RESULTS_DIR, "llm_brief_index.csv"), index=False)

    print(f"[llm] wrote {len(rows)} briefs to {out_md}")
    return rows


if __name__ == "__main__":
    run()
