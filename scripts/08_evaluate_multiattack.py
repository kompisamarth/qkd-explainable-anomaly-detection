"""Evaluate the trained multi-attack XGBoost checkpoint (single seed).

Writes, for the session-held-out TEST split:
  multiattack_overall_benchmarks.csv   multi-class metrics of the model
  multiattack_binary_benchmarks.csv    attack-vs-normal FNR/FPR: model vs fixed 11% vs
                                       finite-size 11% (audit L-3: one task per table)
  multiattack_per_class_f1.csv         per-class precision / recall / F1
  fig_confusion_matrix_5class.png      (name kept for compatibility; now N classes)
  fig_roc_curves_5class.png
and the same metrics on the held-out GENERALISATION set (unseen attack intensities and
distances) as multiattack_heldout_*.csv. Multi-seed statistics with confidence
intervals come from scripts/10_multiseed_evaluation.py.
"""

import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_curve)

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.dataset import model_features
from src.evaluation.metrics import binary_rates, finite_size_threshold_alarm


def load_config() -> dict:
    """config.yaml merged with config/physics.yaml (see src/config.py)."""
    from src.config import load_config as _load
    return _load()


def class_labels(cfg, n):
    if cfg["multiattack"].get("engine", "calibrated") == "legacy":
        return ["Normal Channel", "Fractional IR", "PNS Attack", "Fake State", "Time-Shift"][:n]
    from src.simulator.calibrated import CLASS_NAMES
    return CLASS_NAMES[:n]


def evaluate(df, model, feats, labels, tag, metrics_dir, figures_dir, abort_qber):
    n = len(labels)
    y = df["block_class"].to_numpy(int)
    probs = model.predict_proba(df[feats].values)
    pred = probs.argmax(axis=1)
    prefix = "multiattack" if tag == "test" else "multiattack_heldout"

    overall = pd.DataFrame([{
        "model": "XGBoost multi-attack",
        "split": tag,
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=range(n), zero_division=0),
        "macro_precision": precision_score(y, pred, average="macro", labels=range(n), zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", labels=range(n), zero_division=0),
        "false_negative_rate": binary_rates(y > 0, pred > 0)["fnr"],
    }])
    overall.to_csv(metrics_dir / f"{prefix}_overall_benchmarks.csv", index=False)

    y_bin = y > 0
    qber = df["qber_mean"].to_numpy(float)
    rows = []
    for name, alarm in [
        ("XGBoost (any attack class)", pred > 0),
        (f"Fixed {abort_qber:.0%} QBER threshold (asymptotic)", qber > abort_qber),
        (f"Finite-size {abort_qber:.0%} (Wilson lower bound > threshold)", finite_size_threshold_alarm(df, abort_qber)),
    ]:
        r = binary_rates(y_bin, alarm)
        rows.append({"detector": name, "split": tag, **r})
    pd.DataFrame(rows).to_csv(metrics_dir / f"{prefix}_binary_benchmarks.csv", index=False)

    rep = classification_report(y, pred, labels=range(n), target_names=labels, output_dict=True, zero_division=0)
    per_class = pd.DataFrame([{"class_name": c, "precision": rep[c]["precision"], "recall": rep[c]["recall"],
                               "f1_score": rep[c]["f1-score"], "support": int(rep[c]["support"])} for c in labels])
    per_class.to_csv(metrics_dir / f"{prefix}_per_class_f1.csv", index=False)

    cm = confusion_matrix(y, pred, labels=range(n))
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(metrics_dir / f"{prefix}_confusion_matrix.csv")
    print(f"\n[{tag}] accuracy {overall.accuracy[0]:.4f}  macro-F1 {overall.macro_f1[0]:.4f}  "
          f"FNR {overall.false_negative_rate[0]:.4f}")
    print(per_class.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(rows)[["detector", "fnr", "fpr"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if tag == "test":
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        plt.figure(figsize=(9, 7), dpi=300)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels,
                    cbar_kws={"label": "Row-normalised proportion"})
        plt.title(f"{n}-class confusion matrix (XGBoost, simulated data, session-held-out test)", fontsize=11)
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.xticks(rotation=30, ha="right", fontsize=8); plt.yticks(fontsize=8)
        plt.tight_layout(); plt.savefig(figures_dir / "fig_confusion_matrix_5class.png", dpi=300); plt.close()

        plt.figure(figsize=(8, 6), dpi=300)
        aucs = []
        for c in range(n):
            fpr, tpr, _ = roc_curve(y == c, probs[:, c])
            aucs.append(auc(fpr, tpr))
            plt.plot(fpr, tpr, lw=1.8, label=f"{labels[c]} (AUC {aucs[-1]:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.title(f"One-vs-rest ROC (macro AUC {np.mean(aucs):.3f}; simulated data)")
        plt.legend(fontsize=7.5, loc="lower right"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(figures_dir / "fig_roc_curves_5class.png", dpi=300); plt.close()


def main():
    cfg = load_config()
    data_dir = ROOT_DIR / "data" / "processed"
    metrics_dir = ROOT_DIR / "results" / "metrics"
    figures_dir = ROOT_DIR / "results" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True); figures_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(data_dir / "checkpoints" / "xgb_multiattack.joblib")
    test = pd.read_csv(data_dir / "test_multiattack.csv")
    feats = model_features(test)
    n = int(model.n_classes_) if hasattr(model, "n_classes_") else int(test["block_class"].max()) + 1
    labels = class_labels(cfg, n)
    abort = float(cfg["calibrated_bb84"]["asymptotic_abort_qber"])

    print("=" * 70)
    print(f"EVALUATING XGBoost CHECKPOINT ({n} classes, {len(feats)} features)")
    print("=" * 70)
    evaluate(test, model, feats, labels, "test", metrics_dir, figures_dir, abort)
    held_path = data_dir / "heldout_generalization.csv"
    if held_path.exists():
        evaluate(pd.read_csv(held_path), model, feats, labels, "heldout", metrics_dir, figures_dir, abort)


if __name__ == "__main__":
    main()
