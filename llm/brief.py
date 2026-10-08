"""
brief.py — generate one condition brief: payload → prompt → Ollama → validate → store.

If the LLM output fails validation it is retried once with the errors fed back.
If it fails again, the deterministic template brief is used and the trace says so.
Every attempt (accepted or not) is kept in the trace.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError

from llm import llm_client, template_brief
from llm.payload import Payload, build_payload
from llm.schema import BriefText, render_markdown
from llm.validate import validate

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
TRACE_PATH = os.path.join(OUTPUTS_DIR, "briefs.jsonl")
MAX_ATTEMPTS = 2

_env = Environment(loader=FileSystemLoader(PROMPTS_DIR), keep_trailing_newline=False)


def _view_v1(p: Payload) -> dict:
    return p.for_prompt()


def _view_v2(p: Payload) -> dict:
    """v1 view plus, per section, the figures that section must contain.

    The lists mirror validate.completeness(), so the prompt asks for exactly what
    the validator checks. No new numbers: every entry is already in the section.
    """
    v = p.for_prompt()
    for o in v["outlook"]:
        o["numbers_to_include"] = [o["predicted"], o["typical_error"], o["range_low"],
                                   o["range_high"]]
    if v["repair"] is not None:
        r = v["repair"]
        r["numbers_to_include"] = [r["probability_pct"], r["test_repairs_caught"],
                                   r["test_repairs_total"]]
    return v


# name -> (payload view, example file suffix). Old versions stay so runs are reproducible.
PROMPTS = {
    "brief_v1": (_view_v1, "_example.json"),
    "brief_v2": (_view_v2, "_examples.json"),
}
PROMPT_NAME = os.environ.get("LLM_PROMPT", "brief_v2")


def prompt_files(name: str) -> list[str]:
    return [os.path.join(PROMPTS_DIR, name + s) for s in (".j2", PROMPTS[name][1])]


def _render_prompt(p: Payload, name: str) -> tuple[list[dict], str]:
    """System rules, worked example(s) (user → assistant), then this payload.

    The hash covers the template and the examples, so any change to either is
    visible in the trace.
    """
    view, _ = PROMPTS[name]
    h = hashlib.sha256()
    for path in prompt_files(name):
        with open(path, "rb") as f:
            h.update(f.read())
    with open(prompt_files(name)[1], encoding="utf-8") as f:
        examples = json.load(f)
    if isinstance(examples, dict):  # v1: a single example object; v2+: a list
        examples = [examples]

    tmpl = _env.get_template(f"{name}.j2")

    def user_msg(payload: dict) -> str:
        ctx = tmpl.new_context({"payload_json": json.dumps(payload, indent=1,
                                                           ensure_ascii=False)})
        return "".join(tmpl.blocks["user"](ctx)).strip()

    messages = [{"role": "system",
                 "content": "".join(tmpl.blocks["system"](tmpl.new_context({}))).strip()}]
    for ex in examples:
        messages += [
            {"role": "user", "content": user_msg(ex["payload"])},
            {"role": "assistant", "content": json.dumps(ex["brief"], ensure_ascii=False)},
        ]
    messages.append({"role": "user", "content": user_msg(view(p))})
    return messages, h.hexdigest()[:12]


def generate(p: Payload, model: str = llm_client.DEFAULT_MODEL, use_llm: bool = True,
             store: bool = True, prompt: str | None = None) -> dict:
    """Returns the trace record; record["markdown"] is the brief to show."""
    prompt = prompt or PROMPT_NAME
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "road_id": p.identity.road_id, "survey_date": p.identity.survey_date,
        "payload": p.model_dump(), "prompt": None, "model": None, "attempts": [],
        "source": None, "brief": None, "validation": None, "markdown": None,
    }
    brief, result = None, None

    if use_llm:
        messages, sha = _render_prompt(p, prompt)
        record["prompt"] = {"name": prompt, "sha256_12": sha}
        record["model"] = {"tag": model, "digest": llm_client.model_digest(model),
                           "ollama_version": llm_client.server_version()}
        schema = BriefText.model_json_schema()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            resp = llm_client.chat(messages, schema, model=model)
            entry = {"attempt": attempt, "raw": resp.content, "seconds": resp.seconds,
                     "eval_tokens": resp.eval_tokens, "options": resp.options}
            try:
                candidate = BriefText.model_validate_json(resp.content)
            except ValidationError as e:
                entry["validation"] = {"ok": False, "errors": [f"schema: {e.errors()[:3]}"]}
                record["attempts"].append(entry)
                continue
            res = validate(candidate, p)
            entry["validation"] = vars(res)
            record["attempts"].append(entry)
            if res.ok:
                brief, result = candidate, res
                record["source"] = "llm" if attempt == 1 else "llm_retry"
                break
            messages = messages + [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": "That brief broke the rules: "
                 + "; ".join(res.errors) + ". Rewrite it, using only numbers that appear "
                 "in the payload."}]

    if brief is None:
        brief = template_brief.build(p)
        result = validate(brief, p)
        record["source"] = "template" if not use_llm else "template_fallback"

    record["brief"] = brief.model_dump()
    record["validation"] = vars(result)
    record["markdown"] = render_markdown(brief, p)

    if store:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        with open(TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    return record


def brief_for(road_id: str, survey_date, **kw) -> dict:
    return generate(build_payload(road_id, survey_date), **kw)
