"""
cli.py — command line entry point.

    python -m llm.cli list --split test
    python -m llm.cli payload --road 1-4126-1 --date 1993-03-30
    python -m llm.cli brief   --road 1-4126-1 --date 1993-03-30 [--model qwen3:8b] [--no-llm]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="llm.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="list available surveys")
    s.add_argument("--split", choices=["train", "test"])

    for name in ("payload", "brief"):
        s = sub.add_parser(name)
        s.add_argument("--road", required=True, help="ROAD_ID, e.g. 1-4126-1")
        s.add_argument("--date", required=True, help="survey date, yyyy-mm-dd or d/m/yyyy")
        if name == "brief":
            from llm.llm_client import DEFAULT_MODEL
            from llm.brief import PROMPT_NAME, PROMPTS
            s.add_argument("--model", default=DEFAULT_MODEL)
            s.add_argument("--prompt", choices=sorted(PROMPTS), default=PROMPT_NAME)
            s.add_argument("--no-llm", action="store_true", help="template brief only")
            s.add_argument("--no-store", action="store_true")
            s.add_argument("--show-hindsight", action="store_true",
                           help="print the observed next-survey values after the brief")

    a = ap.parse_args(argv)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    from llm.payload import build_payload, list_surveys

    if a.cmd == "list":
        print(list_surveys(a.split).to_string(index=False))
        return 0

    p = build_payload(a.road, a.date)
    if a.cmd == "payload":
        print(json.dumps(p.model_dump(), indent=1))
        return 0

    from llm.brief import generate
    rec = generate(p, model=a.model, use_llm=not a.no_llm, store=not a.no_store,
                   prompt=a.prompt)
    print(rec["markdown"])
    v = rec["validation"]
    print(f"--\nprompt: {a.prompt if not a.no_llm else '-'}   "
          f"source: {rec['source']}   attempts: {len(rec['attempts'])}   "
          f"validation ok: {v['ok']}   numbers checked: {v['numbers_checked']}")
    for w in v["warnings"]:
        print(f"warning: {w}")
    for att in rec["attempts"]:
        if not att["validation"]["ok"]:
            print(f"attempt {att['attempt']} rejected: {att['validation']['errors']}")
    if a.show_hindsight:
        print(f"hindsight: {p.hindsight.model_dump()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
