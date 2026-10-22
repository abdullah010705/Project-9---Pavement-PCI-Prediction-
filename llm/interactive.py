"""
interactive.py — pick a road and a survey date by answering prompts.

Added after the client asked for the layer to take user input directly. The CLI already
accepted --road and --date, but you had to know a valid pair; this asks instead, and only
offers combinations that exist.

    ./run_llm.sh                 (no arguments)
    python -m llm.interactive

Nothing about how a brief is produced changes — this only chooses the inputs. The facts
are still built in code and every number is still validated.
"""
from __future__ import annotations

import os
import sys
import warnings

# The models were pickled by an earlier XGBoost; loading them prints a long warning that
# looks like a failure in the middle of a demo. The models are verified by the test suite
# to reproduce the reported numbers, so the warning is noise here.
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")


def _ask(prompt: str, valid, allow_blank_default=None):
    """Ask until the answer is in `valid`. Blank returns the default if one is given."""
    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            sys.exit(0)
        if not raw and allow_blank_default is not None:
            return allow_blank_default
        if raw.lower() in {"q", "quit", "exit"}:
            print("Cancelled.")
            sys.exit(0)
        if raw in valid:
            return raw
        print(f"  Not one of the options. Enter a number 1-{len(valid)}, or q to quit.")


def choose(split: str = "test"):
    """Returns (road_id, survey_date_iso) chosen by the user."""
    from llm.payload import list_surveys

    surveys = list_surveys(split)
    roads = sorted(surveys.ROAD_ID.unique())

    print(f"\nRoads available in the {split} set ({len(roads)} of them)\n")
    for i, r in enumerate(roads, 1):
        rows = surveys[surveys.ROAD_ID == r]
        worst = rows.pci.min()
        if worst != worst:                      # NaN: this road has no PCI recorded
            print(f"  {i:>3}. {r:<12} {len(rows)} survey(s), lowest PCI     -")
        else:
            flag = "  <- poor condition" if worst < 55 else ""
            print(f"  {i:>3}. {r:<12} {len(rows)} survey(s), lowest PCI {worst:>5.1f}{flag}")

    pick = _ask(f"\nChoose a road (1-{len(roads)}), or q to quit: ",
                {str(i) for i in range(1, len(roads) + 1)})
    road = roads[int(pick) - 1]

    dates = surveys[surveys.ROAD_ID == road].sort_values("SURVEY_DATE_ISO")
    print(f"\nSurveys for road {road}\n")
    for i, (_, row) in enumerate(dates.iterrows(), 1):
        print(f"  {i:>3}. {row.SURVEY_DATE_ISO}   PCI {row.pci:>5.1f}   "
              f"predicts: {row.targets}")

    if len(dates) == 1:
        print("\n  Only one survey — using it.")
        date = dates.iloc[0].SURVEY_DATE_ISO
    else:
        pick = _ask(f"\nChoose a survey (1-{len(dates)}) [default 1], or q to quit: ",
                    {str(i) for i in range(1, len(dates) + 1)}, allow_blank_default="1")
        date = dates.iloc[int(pick) - 1].SURVEY_DATE_ISO

    return road, date


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Choose a road and survey, then write a brief.")
    ap.add_argument("--split", default="test", choices=["train", "test"],
                    help="which split to offer (default: test — roads the models never saw)")
    ap.add_argument("--model", default=None, help="Ollama model tag")
    ap.add_argument("--no-llm", action="store_true", help="use the template, skip Ollama")
    a = ap.parse_args(argv)

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    road, date = choose(a.split)

    print(f"\nWriting a brief for road {road}, survey {date} ...")
    if not a.no_llm:
        print("(the language model takes about 30-40 seconds)\n")

    from llm.brief import generate
    from llm.payload import build_payload
    from llm import llm_client
    from llm.schema import render_markdown

    p = build_payload(road, date)
    rec = generate(p, model=a.model or llm_client.DEFAULT_MODEL, use_llm=not a.no_llm)

    print(rec["markdown"])
    v = rec.get("validation") or {}
    print(f"\n--\nsource: {rec.get('source')}   attempts: {len(rec.get('attempts') or [])}"
          f"   validation ok: {v.get('ok')}   numbers checked: {v.get('numbers_checked')}")

    try:
        again = input("\nAnother road? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return 0
    if again.startswith("y"):
        return main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
