"""
llm_client.py — thin wrapper around a local Ollama server.

Settings for reproducibility: temperature 0, fixed seed, and a JSON schema passed
as `format` so output is structured. The model tag, digest and Ollama version are
recorded with every call.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache

import ollama

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "llama3.1:8b")
DEFAULT_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192}


@dataclass
class LLMResponse:
    content: str
    model: str
    model_digest: str | None
    options: dict
    seconds: float
    eval_tokens: int | None


@lru_cache(maxsize=None)
def model_digest(model: str) -> str | None:
    try:
        for m in ollama.list().models:
            if m.model == model:
                return m.digest
    except Exception:
        return None
    return None


def server_version() -> str | None:
    try:
        import httpx
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        return httpx.get(f"{host}/api/version", timeout=5).json().get("version")
    except Exception:
        return None


def chat(messages: list[dict], schema: dict, model: str = DEFAULT_MODEL,
         options: dict | None = None) -> LLMResponse:
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    kwargs = {}
    if model.startswith("qwen3"):
        kwargs["think"] = False  # qwen3 reasons by default; we want the answer only
    t0 = time.perf_counter()
    r = ollama.chat(model=model, messages=messages, format=schema, options=opts, **kwargs)
    return LLMResponse(content=r.message.content, model=model,
                       model_digest=model_digest(model), options=opts,
                       seconds=round(time.perf_counter() - t0, 2),
                       eval_tokens=getattr(r, "eval_count", None))
