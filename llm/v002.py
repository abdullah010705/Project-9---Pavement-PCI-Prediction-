"""
v002.py — the only module that reaches into the pipeline.

Naming note: the file name and the V002_DIR environment variable keep their
original spelling even though the pipeline folder is now V0.0.3. They are code
identifiers, not version labels — renaming them would break Luke's tests, his
evaluation runner and anyone's existing V002_DIR override, for no benefit.

Everything the LLM layer needs from the team pipeline goes through here: config,
the modelling datasets, the exact feature matrix the models were trained on, the
fitted models, and the results files. Read-only: nothing is written into the pipeline.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

LLM_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(LLM_DIR)


def _default_v002_dir() -> str:
    """Locate the pipeline.

    Two layouts are supported. In Luke's original delivery `llm/` sits NEXT TO a
    pipeline folder. After integration `llm/` lives INSIDE the pipeline, so the parent
    directory is the pipeline itself. Detect by looking for config.py.
    Override either with the V002_DIR environment variable.
    """
    if os.path.exists(os.path.join(WORKSPACE_DIR, "config.py")):
        return WORKSPACE_DIR                       # llm/ is inside the pipeline
    for name in ("V0.0.3", "V0.0.2"):          # llm/ sitting beside the pipeline
        nested = os.path.join(WORKSPACE_DIR, name)
        if os.path.exists(os.path.join(nested, "config.py")):
            return nested
    return os.path.join(WORKSPACE_DIR, "V0.0.3")


V002_DIR = os.environ.get("V002_DIR", _default_v002_dir())

# Keep the team folder clean: no __pycache__ written into the pipeline from here.
sys.dont_write_bytecode = True
if V002_DIR not in sys.path:
    sys.path.insert(0, V002_DIR)

import config  # noqa: E402  (the pipeline's config.py)
from src.data.pci_algorithm_v2_reference import (  # noqa: E402,F401
    classify_rut_severity, rating_from_pci)
from src.models.feature_reduction import build_matrix  # noqa: E402

TARGET_KEYS = tuple(config.TARGETS)  # ("pci", "rut")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def dataset(target_key: str) -> pd.DataFrame:
    """The modelling dataset exactly as the pipeline reads it (same row order)."""
    return pd.read_csv(config.TARGETS[target_key]["dataset_path"],
                       low_memory=False).reset_index(drop=True)


@lru_cache(maxsize=None)
def reduced_features(target_key: str) -> list[str]:
    path = os.path.join(config.RESULTS_DIR, f"reduced_features_{target_key}.json")
    with open(path) as f:
        return json.load(f)["features"]


@lru_cache(maxsize=None)
def feature_matrix(target_key: str) -> pd.DataFrame:
    """Rebuild the model input matrix the same way xgboost_shap.run() does.

    Imputation and one-hot levels are fitted on training rows only (build_matrix),
    then the reduced feature set is selected in the order stored in the JSON.
    """
    t = config.TARGETS[target_key]
    df = dataset(target_key)
    train_mask = df[config.SPLIT_COL] == config.TRAIN_VALUE
    numeric = [c for c in t["numeric_feature_cols"] if c in df.columns]
    categorical = [c for c in t["categorical_feature_cols"] if c in df.columns]
    X = build_matrix(df, numeric, categorical, train_mask)
    return X[reduced_features(target_key)]


@lru_cache(maxsize=None)
def raw_master() -> pd.DataFrame:
    cols = ["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO", "SURVEY_DATE",
            "PCI_SCORE", "PCI_ERROR", "LAST_PATCH_DATE"]
    df = pd.read_csv(config.RAW_DATA_PATH, low_memory=False,
                     usecols=lambda c: c in cols)
    return df


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class _HandleCapture(pickle.Unpickler):
    """Unpickle an XGBoost sklearn wrapper without running Booster.__setstate__.

    The shipped joblibs were written on macOS. Their serialised learner *config*
    does not load under the Windows xgboost build ("input stream corrupted" /
    "Invalid cast, from Null to Object"), but the *model* section — the trees,
    base score, objective and feature names — is intact. We capture the raw
    buffer and load only the model section. No refitting is involved; the
    loaded model is checked against the reported test metrics in tests/.

    Pickle can execute code. These files are the team's own pipeline artefacts,
    and find_class only allows the few modules an XGBoost wrapper needs.
    """

    _ALLOWED = ("xgboost.", "numpy", "builtins", "collections", "copyreg")

    def find_class(self, module, name):
        if module == "xgboost.core" and name == "Booster":
            return _RawBooster
        if not module.startswith(self._ALLOWED):
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        return super().find_class(module, name)


class _RawBooster:
    def __setstate__(self, state):
        self.raw = bytes(state["handle"]) if state.get("handle") is not None else None


def _model_section(ubj: bytes) -> bytearray:
    """Extract the value of the top-level "Model" key from an XGBoost UBJSON blob."""
    key = b"L\x00\x00\x00\x00\x00\x00\x00\x05Model{"
    i = ubj.find(key)
    if i < 0:
        raise ValueError("no Model section found in booster buffer")
    # The Model object runs to the byte before the final closing brace.
    return bytearray(ubj[i + len(key) - 1:-1])


@lru_cache(maxsize=None)
def booster(name: str):
    """Load models/<name>.joblib as an xgboost.Booster.

    name: "xgboost_pci", "xgboost_rut" or "repair_classifier".
    """
    import joblib
    import xgboost as xgb

    path = os.path.join(config.MODELS_DIR, f"{name}.joblib")
    try:
        return joblib.load(path).get_booster()
    except Exception:
        pass
    with open(path, "rb") as f:
        wrapper = _HandleCapture(io.BytesIO(f.read())).load()
    bst = xgb.Booster()
    bst.load_model(_model_section(wrapper._Booster.raw))
    return bst


def predict(name: str, X: pd.DataFrame, contribs: bool = False) -> np.ndarray:
    """Model output for rows of X (probability for the classifier).

    contribs=True returns per-feature SHAP contributions (TreeSHAP, XGBoost native)
    with the bias term in the last column, in the model's raw output space.
    """
    import xgboost as xgb
    bst = booster(name)
    if list(bst.feature_names) != list(X.columns):
        raise ValueError(f"{name}: feature order mismatch with reduced_features JSON")
    dm = xgb.DMatrix(X, feature_names=list(X.columns))
    return bst.predict(dm, pred_contribs=contribs)


# ---------------------------------------------------------------------------
# Results files (the only source for error figures)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def results_json(name: str) -> dict:
    with open(os.path.join(config.RESULTS_DIR, name)) as f:
        return json.load(f)


def model_error(target_key: str) -> dict:
    """Test and grouped-CV error for tuned XGBoost, plus the bootstrap verdict."""
    comp = results_json("comparison_vs_baselines.json")[target_key]
    xgb_row = next(r for r in comp["table"] if r["model"] == "XGBoost (tuned)")
    return {
        "test_rmse": xgb_row["test_rmse"], "test_mae": xgb_row["test_mae"],
        "cv_rmse": xgb_row["cv_rmse"], "cv_mae": xgb_row["cv_mae"],
        "bootstrap_verdict": comp["bootstrap"]["verdict"],
    }


def classifier_summary() -> dict:
    return results_json("repair_classifier_summary.json")
