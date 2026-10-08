"""
check_env.py — confirm everything the LLM layer needs is in place.

    python -m llm.check_env                     # full check, default model
    python -m llm.check_env --model qwen3:8b
    python -m llm.check_env --packages-only     # used by run_llm.ps1 before installing
    python -m llm.check_env --no-ollama         # skip Ollama checks (template-only runs)

Exit code 0 = all required checks passed (warnings allowed), 1 = something is missing.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

LLM_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(LLM_DIR)
REQUIREMENTS = os.path.join(LLM_DIR, "requirements.txt")

_fails, _warns = [], []


def ok(msg):
    print(f"  [ OK ] {msg}")


def warn(msg):
    _warns.append(msg)
    print(f"  [WARN] {msg}")


def fail(msg):
    _fails.append(msg)
    print(f"  [FAIL] {msg}")


def check_python():
    print("Python")
    v = sys.version_info
    if v < (3, 10):
        fail(f"Python {v.major}.{v.minor} — V0.0.3 needs 3.10 or later")
    elif (v.major, v.minor) != (3, 12):
        warn(f"Python {v.major}.{v.minor}.{v.micro} — team environment is 3.12")
    else:
        ok(f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})")


def check_packages():
    print("Packages (llm/requirements.txt)")
    from importlib.metadata import PackageNotFoundError, version
    extra = ["matplotlib", "seaborn", "httpx"]  # needed by V0.0.3 imports / Ollama client
    pins = {}
    with open(REQUIREMENTS) as f:
        for line in f:
            m = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*==\s*([^\s#]+)", line)
            if m:
                pins[m.group(1)] = m.group(2)
    for name in extra:
        pins.setdefault(name, None)
    for name, want in pins.items():
        try:
            have = version(name)
        except PackageNotFoundError:
            fail(f"{name} not installed")
            continue
        if want and have != want:
            warn(f"{name} {have} (verified with {want}) — numbers may differ")
        else:
            ok(f"{name} {have}")


def check_v002():
    print("V0.0.3 (read-only)")
    try:
        from llm import v002
    except Exception as e:  # noqa: BLE001
        fail(f"cannot import V0.0.3 config: {e}")
        return
    ok(f"found {v002.V002_DIR}")
    cfg = v002.config
    for path in [cfg.MODEL_DATASET_PCI_PATH, cfg.MODEL_DATASET_RUT_PATH,
                 cfg.SPLIT_ASSIGNMENT_PATH, cfg.RAW_DATA_PATH]:
        (ok if os.path.exists(path) else fail)(f"data/{os.path.basename(path)}")
    for name in ["comparison_vs_baselines.json", "repair_classifier_summary.json",
                 "reduced_features_pci.json", "reduced_features_rut.json"]:
        (ok if os.path.exists(os.path.join(cfg.RESULTS_DIR, name)) else fail)(f"results/{name}")

    import numpy as np
    for key in v002.TARGET_KEYS:
        try:
            df, X = v002.dataset(key), v002.feature_matrix(key)
            test = df[cfg.SPLIT_COL] == cfg.TEST_VALUE
            pred = v002.predict(f"xgboost_{key}", X[test])
            y = df.loc[test, cfg.TARGETS[key]["target_col"]].values
            rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
            want = v002.model_error(key)["test_rmse"]
            if abs(rmse - want) < 5e-4:
                ok(f"models/xgboost_{key} loads, test RMSE {rmse:.3f} matches results/")
            else:
                fail(f"models/xgboost_{key} test RMSE {rmse:.3f} ≠ results/ {want:.3f} — "
                     "models and results are out of sync (re-run pipeline?)")
        except Exception as e:  # noqa: BLE001
            fail(f"models/xgboost_{key}: {e}")
    try:
        X = v002.feature_matrix("pci")
        v002.predict("repair_classifier", X.iloc[:1])
        ok("models/repair_classifier loads")
    except Exception as e:  # noqa: BLE001
        fail(f"models/repair_classifier: {e}")


def check_llm_files():
    print("LLM layer files")
    from llm.brief import PROMPT_NAME, PROMPTS, prompt_files
    if PROMPT_NAME not in PROMPTS:
        fail(f"LLM_PROMPT={PROMPT_NAME} is not one of {sorted(PROMPTS)}")
        return
    for p in prompt_files(PROMPT_NAME):
        (ok if os.path.exists(p) else fail)(f"prompts/{os.path.basename(p)} (active prompt)")
    sample = os.path.join(LLM_DIR, "eval", "sample.csv")
    (ok if os.path.exists(sample) else warn)("eval/sample.csv" + (
        "" if os.path.exists(sample) else " missing — run: python -m llm.eval.run_eval sample"))


def check_ollama(model: str):
    print(f"Ollama (model {model})")
    try:
        from llm import llm_client
        v = llm_client.server_version()
        if not v:
            fail("Ollama server not reachable at "
                 + os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
            return
        ok(f"server running, version {v}")
        digest = llm_client.model_digest(model)
        if digest:
            ok(f"{model} available ({digest[:12]})")
        else:
            fail(f"{model} not pulled — run: ollama pull {model}")
    except Exception as e:  # noqa: BLE001
        fail(f"Ollama check failed: {e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="llm.check_env")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "llama3.1:8b"))
    ap.add_argument("--packages-only", action="store_true")
    ap.add_argument("--no-ollama", action="store_true")
    a = ap.parse_args(argv)

    check_python()
    check_packages()
    if not a.packages_only and not _fails:
        check_v002()
        check_llm_files()
        if not a.no_ollama:
            check_ollama(a.model)

    print(f"\n{len(_fails)} failed, {len(_warns)} warnings")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
