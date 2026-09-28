"""
repair_classifier.py — "does this road need repair?" (owner: Abdullah).

The client's decision rule: PCI below 55 at the next survey means the road needs repair.
This predicts that directly rather than thresholding a regression, because the decision
is what an asset manager acts on.

PCI dataset only — NEEDS_REPAIR does not exist for rutting.

Reported with precision, recall, F1 and a confusion matrix, not just accuracy. The
classes are imbalanced (~22% positive), so accuracy alone flatters the model: predicting
"no repair" for everything already scores ~78%.

Missing a road that needs repair (a false negative) costs more than inspecting one that
does not. The threshold sweep shows that trade-off explicitly so the client can choose
an operating point rather than accepting 0.5 by default.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

import config
import results_io
from src.models.feature_reduction import build_matrix
from src.models.xgboost_shap import load_reduced

N_FOLDS = 5


def get_clf(params=None, scale_pos_weight=1.0):
    from xgboost import XGBClassifier
    defaults = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
                    colsample_bytree=0.8, objective="binary:logistic",
                    eval_metric="logloss", tree_method="hist",
                    scale_pos_weight=scale_pos_weight,
                    random_state=config.RANDOM_SEED,
                    n_jobs=int(os.environ.get("PCI_N_JOBS", 4)))
    defaults.update(params or {})
    return XGBClassifier(**defaults)


def _scores(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run():
    spec = config.CLASSIFIER
    df = pd.read_csv(spec["dataset_path"], low_memory=False).reset_index(drop=True)
    target_col = spec["target_col"]

    train_mask = df[config.SPLIT_COL] == config.TRAIN_VALUE
    y = df[target_col].astype(int)
    groups = df[config.PHYSICAL_ROAD_ID_COL]

    t = config.TARGETS["pci"]
    numeric = [c for c in t["numeric_feature_cols"] if c in df.columns]
    categorical = [c for c in t["categorical_feature_cols"] if c in df.columns]
    X_full = build_matrix(df, numeric, categorical, train_mask)
    reduced = load_reduced("pci")
    X = X_full[reduced] if reduced and all(c in X_full.columns for c in reduced) else X_full

    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[~train_mask], y[~train_mask]

    pos_rate = float(ytr.mean())
    spw = (1 - pos_rate) / pos_rate          # counteract the class imbalance
    print(f"[repair] threshold PCI < {config.REPAIR_THRESHOLD}   features {X.shape[1]}")
    print(f"[repair] train {len(ytr)} ({ytr.sum()} positive, {100*pos_rate:.1f}%)   "
          f"test {len(yte)} ({yte.sum()} positive)")
    print(f"[repair] always-predict-no-repair accuracy on test: {100*(1-yte.mean()):.1f}%")

    clf = get_clf(scale_pos_weight=spw)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    s = _scores(yte, pred, prob)

    print(f"[repair] accuracy {s['accuracy']:.3f}  precision {s['precision']:.3f}  "
          f"recall {s['recall']:.3f}  F1 {s['f1']:.3f}  ROC-AUC {s['roc_auc']:.3f}  "
          f"PR-AUC {s['pr_auc']:.3f}")
    print(f"[repair] confusion matrix: TN {s['tn']}  FP {s['fp']}  FN {s['fn']}  TP {s['tp']}")
    print(f"[repair] of {s['tp']+s['fn']} roads that needed repair, "
          f"{s['tp']} were caught and {s['fn']} were missed")

    # --- grouped CV over every row -------------------------------------------
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))
    oof = np.full(len(X), np.nan)
    for tr_i, te_i in folds:
        c = get_clf(scale_pos_weight=spw)
        c.fit(X.iloc[tr_i], y.iloc[tr_i])
        oof[te_i] = c.predict_proba(X.iloc[te_i])[:, 1]
    s_cv = _scores(y, (oof >= 0.5).astype(int), oof)
    print(f"[repair] grouped CV, all {len(y)} rows: accuracy {s_cv['accuracy']:.3f}  "
          f"precision {s_cv['precision']:.3f}  recall {s_cv['recall']:.3f}  "
          f"F1 {s_cv['f1']:.3f}  PR-AUC {s_cv['pr_auc']:.3f}")

    # --- threshold sweep: the cost trade-off the client should choose from ----
    sweep = []
    for thr in np.arange(0.1, 0.91, 0.05):
        p = (oof >= thr).astype(int)
        sweep.append({"threshold": round(float(thr), 2), **_scores(y, p, oof)})
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(os.path.join(config.RESULTS_DIR, "repair_threshold_sweep.csv"),
                    index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(sweep_df.threshold, sweep_df.precision, "o-", label="Precision", color="#1f3b63")
    ax.plot(sweep_df.threshold, sweep_df.recall, "s-", label="Recall", color="#c0392b")
    ax.plot(sweep_df.threshold, sweep_df.f1, "^-", label="F1", color="#888")
    ax.axvline(0.5, ls=":", lw=1, color="#555")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score")
    ax.set_title(f"Repair classifier — precision/recall trade-off (PCI < {config.REPAIR_THRESHOLD})")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "repair_threshold_sweep.png"), dpi=150)
    plt.close(fig)

    cm = np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["No repair", "Repair"]); ax.set_yticks([0, 1], ["No repair", "Repair"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Repair classifier — test set")
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "repair_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    results_io.append_classifier_results(spec["results_metrics_path"], [
        {"model": f"XGBoost classifier (PCI < {config.REPAIR_THRESHOLD})",
         "precision": s["precision"], "recall": s["recall"], "f1": s["f1"],
         "tp": s["tp"], "fp": s["fp"], "fn": s["fn"], "tn": s["tn"]},
    ])
    with open(os.path.join(config.RESULTS_DIR, "repair_classifier_summary.json"), "w") as f:
        json.dump({"threshold_pci": config.REPAIR_THRESHOLD, "test": s,
                   "grouped_cv_all_rows": s_cv,
                   "train_positive_rate": pos_rate,
                   "majority_class_accuracy_test": float(1 - yte.mean()),
                   "n_features": int(X.shape[1])}, f, indent=2)

    import joblib
    joblib.dump(clf, os.path.join(config.MODELS_DIR, "repair_classifier.joblib"))
    return s


if __name__ == "__main__":
    run()
