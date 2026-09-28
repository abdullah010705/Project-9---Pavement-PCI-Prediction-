"""
reconcile_baseline.py — find where two anchored-trend implementations diverge.

Luke reports the anchored age-trend baseline at R2 0.802 / RMSE 12.05 on PCI_NEXT.
The reimplementation in baseline_traditional.py scores 0.708 / 14.61 on the same rows.
Persistence matches to four decimal places across both, so the data, split and metric
code are identical — the difference is in how the curve is built and anchored.

This runs several plausible variants of "anchored age trend" on identical rows and
reports each one's score, so whichever matches 0.802 tells us what Luke actually did.
Once his script lands, drop his predictions in via --luke-predictions and this will
diff them row by row instead of guessing.

    python -m src.models.reconcile_baseline
    python -m src.models.reconcile_baseline --luke-predictions path/to/his_predictions.csv

Variants tested
---------------
  A  curve fitted on TRAIN rows only, anchored: current + (curve(age+gap) - curve(age))
     — what baseline_traditional.py does now
  B  same, but curve fitted on ALL rows (train + test)         <- common accidental leak
  C  curve fitted on the TARGET (PCI_NEXT) rather than current condition
  D  linear-in-age curve rather than log/power-law
  E  ratio anchoring: current * (curve(age+gap) / curve(age))
  F  no time gap — anchor to the curve's value at current age only
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import config

CURRENT = {"pci": "PCI_SCORE", "rut": "RUT_DEPTH_1_8IN"}


def _metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    err = y - p
    ss_res = float(np.sum(err ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"r2": float(1 - ss_res / ss_tot), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err)))}


def _fit(age, val, kind="log"):
    ok = (~np.isnan(age)) & (~np.isnan(val)) & (age > 0)
    a, v = np.asarray(age)[ok], np.asarray(val)[ok]
    if len(a) < 10:
        return lambda x: np.full_like(np.asarray(x, float), float(np.nanmean(v)))
    if kind == "linear":
        c = np.polyfit(a, v, 1)
        return lambda x: np.polyval(c, np.asarray(x, float))
    c = np.polyfit(np.log(a), v, 1)
    return lambda x: np.polyval(c, np.log(np.clip(np.asarray(x, float), 0.01, None)))


def run(target_key="pci", luke_predictions=None):
    t = config.TARGETS[target_key]
    df = pd.read_csv(t["dataset_path"], low_memory=False).reset_index(drop=True)
    cur = CURRENT[target_key]
    tgt = t["target_col"]

    train = df[df[config.SPLIT_COL] == config.TRAIN_VALUE]
    test = df[df[config.SPLIT_COL] == config.TEST_VALUE]
    y = test[tgt].values
    age = test["PAVEMENT_AGE_YEARS"].values
    gap = test["YEARS_TO_NEXT_SURVEY"].fillna(train["YEARS_TO_NEXT_SURVEY"].median()).values
    lo, hi = (0.0, 100.0) if target_key == "pci" else (0.0, None)
    clip = (lambda p: np.clip(p, lo, hi)) if hi else (lambda p: np.clip(p, lo, None))

    variants = {}

    c_tr = _fit(train["PAVEMENT_AGE_YEARS"].values, train[cur].values)
    variants["A  train-only curve, additive anchor (current impl.)"] = \
        clip(test[cur].values + (c_tr(age + gap) - c_tr(age)))

    c_all = _fit(df["PAVEMENT_AGE_YEARS"].values, df[cur].values)
    variants["B  curve fitted on ALL rows (possible leak)"] = \
        clip(test[cur].values + (c_all(age + gap) - c_all(age)))

    c_tgt = _fit(train["PAVEMENT_AGE_YEARS"].values, train[tgt].values)
    variants["C  curve fitted on the target, not current"] = \
        clip(test[cur].values + (c_tgt(age + gap) - c_tgt(age)))

    c_lin = _fit(train["PAVEMENT_AGE_YEARS"].values, train[cur].values, kind="linear")
    variants["D  linear-in-age curve"] = \
        clip(test[cur].values + (c_lin(age + gap) - c_lin(age)))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(c_tr(age)) > 1e-6, c_tr(age + gap) / c_tr(age), 1.0)
    variants["E  ratio anchoring"] = clip(test[cur].values * ratio)

    variants["F  no time gap in the anchor"] = clip(test[cur].values * 0 + c_tr(age))

    variants["   (reference) persistence"] = clip(test[cur].values)

    rows = []
    for name, pred in variants.items():
        rows.append({"variant": name, **{k: round(v, 3) for k, v in _metrics(y, pred).items()}})

    if luke_predictions and os.path.exists(luke_predictions):
        lp = pd.read_csv(luke_predictions)
        col = next((c for c in lp.columns if "ANCHOR" in c.upper()), None)
        if col:
            rows.append({"variant": f"   LUKE ({col})",
                         **{k: round(v, 3) for k, v in _metrics(y, lp[col].values).items()}})
            best = min(variants.items(),
                       key=lambda kv: np.mean(np.abs(kv[1] - lp[col].values)))
            print(f"\nClosest variant to Luke's predictions: {best[0]}  "
                  f"(mean abs difference {np.mean(np.abs(best[1] - lp[col].values)):.4f})")

    out = pd.DataFrame(rows).sort_values("r2", ascending=False)
    print(f"\n=== {target_key.upper()}: anchored-trend variants on the same {len(y)} test rows ===")
    print(out.to_string(index=False))
    print(f"\nLuke reports R2 0.802 / RMSE 12.05 for PCI. "
          f"Whichever variant lands there is what his code does.")
    out.to_csv(os.path.join(config.RESULTS_DIR,
               f"reconcile_baseline_{target_key}.csv"), index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="pci")
    ap.add_argument("--luke-predictions", default=None,
                    help="CSV of Luke's test predictions, to diff row by row")
    a = ap.parse_args()
    run(a.target, a.luke_predictions)
