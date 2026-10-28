"""Phase 2: Academic evaluation, metrics generation, and publication figures.

Produces:
- Table II: Overall Multi-Class Performance (Acc, Macro-F1, Precision, FNR vs 11% Threshold)
- Table III: Per-Class Precision, Recall, and F1-Scores across all 5 classes
- Latency: Detection Latency Delta_T (blocks from onset to true alarm)
- Figure: 5x5 Normalized Confusion Matrix Heatmap (300 DPI)
- Figure: One-vs-Rest ROC Curves across all 5 classes with Macro-AUC (300 DPI)
"""

import sys
import json
import time
from pathlib import Path
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, classification_report
)

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.features.time_series import FEATURE_NAMES_63


CLASS_LABELS = [
    "Normal Channel",
    "Fractional IR",
    "PNS Attack",
    "Fake State",
    "Time-Shift",
]


def load_config() -> dict:
    with open(ROOT_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    data_dir = ROOT_DIR / "data" / "processed"
    test_path = data_dir / "test_multiattack.csv"
    ckpt_path = data_dir / "checkpoints" / "xgb_multiattack.joblib"

    metrics_dir = ROOT_DIR / "results" / "metrics"
    figures_dir = ROOT_DIR / "results" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 2: EVALUATION & PUBLICATION BENCHMARKS (5 CLASSES)")
    print(f"  Test path:   {test_path}")
    print(f"  Model path:  {ckpt_path}")
    print("=" * 70)

    print("\nLoading test dataset and model checkpoint...")
    test_df = pd.read_csv(test_path)
    model = joblib.load(ckpt_path)

    X_test = test_df[FEATURE_NAMES_63].values
    y_test = test_df["block_class"].values.astype(int)

    # Predictions
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    # -------------------------------------------------------------
    # 1. Overall Performance & Comparison vs. 11% Fixed Threshold
    # -------------------------------------------------------------
    print("\n[1/4] Computing Overall Multi-Class Performance & FNR vs. 11% Baseline...")
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    macro_prec = precision_score(y_test, preds, average="macro")
    macro_rec = recall_score(y_test, preds, average="macro")

    # Binary attack-vs-normal evaluation for False Negative Rate (FNR)
    # y_binary: 0 = Normal, 1 = Attack (classes 1, 2, 3, 4)
    y_binary = (y_test > 0).astype(int)
    preds_binary_xgb = (preds > 0).astype(int)

    # Conventional Fixed 11% Threshold baseline: flags attack if qber_mean >= 0.11
    qber_col = test_df["qber_mean"].values if "qber_mean" in test_df.columns else test_df["qber_current"].values
    preds_binary_thresh = (qber_col >= 0.11).astype(int)

    # Compute FNR = FN / (FN + TP)
    def compute_fnr(y_true, y_pred):
        attack_mask = (y_true == 1)
        if np.sum(attack_mask) == 0:
            return 0.0
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return float(fn / (fn + tp))

    fnr_xgb = compute_fnr(y_binary, preds_binary_xgb)
    fnr_thresh = compute_fnr(y_binary, preds_binary_thresh)
    acc_thresh = accuracy_score(y_binary, preds_binary_thresh)

    overall_results = [
        {
            "model": "Conventional Fixed 11% QBER Threshold",
            "accuracy": float(acc_thresh),
            "macro_f1": float(f1_score(y_binary, preds_binary_thresh, average="macro")),
            "macro_precision": float(precision_score(y_binary, preds_binary_thresh, average="macro")),
            "macro_recall": float(recall_score(y_binary, preds_binary_thresh, average="macro")),
            "false_negative_rate": float(fnr_thresh),
        },
        {
            "model": "Proposed XGBoost Multi-Attack Framework",
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "macro_precision": float(macro_prec),
            "macro_recall": float(macro_rec),
            "false_negative_rate": float(fnr_xgb),
        },
    ]

    overall_df = pd.DataFrame(overall_results)
    overall_csv = metrics_dir / "multiattack_overall_benchmarks.csv"
    overall_df.to_csv(overall_csv, index=False)
    print(overall_df.to_string(index=False))

    # -------------------------------------------------------------
    # 2. Per-Class Precision, Recall, and F1-Scores (Table III)
    # -------------------------------------------------------------
    print("\n[2/4] Generating Per-Class Performance Breakdown...")
    rep_dict = classification_report(y_test, preds, target_names=CLASS_LABELS, output_dict=True)
    per_class_rows = []
    for cls_name in CLASS_LABELS:
        m = rep_dict[cls_name]
        per_class_rows.append({
            "class_name": cls_name,
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "f1_score": float(m["f1-score"]),
            "support": int(m["support"]),
        })

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv = metrics_dir / "multiattack_per_class_f1.csv"
    per_class_df.to_csv(per_class_csv, index=False)
    print(per_class_df.to_string(index=False))

    # -------------------------------------------------------------
    # 3. Figure: 5x5 Normalized Confusion Matrix Heatmap (Figure 4)
    # -------------------------------------------------------------
    print("\n[3/4] Generating 5x5 Normalized Confusion Matrix Heatmap (300 DPI)...")
    cm = confusion_matrix(y_test, preds, labels=list(range(5)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6.5), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        cbar_kws={"label": "Normalized Proportion"},
    )
    plt.title("5-Class BB84 Attack Classification Confusion Matrix (XGBoost)", fontweight="bold", fontsize=12)
    plt.xlabel("Predicted Class", fontweight="bold", fontsize=11)
    plt.ylabel("True Class", fontweight="bold", fontsize=11)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_fig_path = figures_dir / "fig_confusion_matrix_5class.png"
    plt.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix to: {cm_fig_path}")

    # -------------------------------------------------------------
    # 4. Figure: One-vs-Rest ROC Curves (Figure 2)
    # -------------------------------------------------------------
    print("\n[4/4] Generating One-vs-Rest ROC Curves (300 DPI)...")
    plt.figure(figsize=(8, 6), dpi=300)
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#9467bd", "#ff7f0e"]

    roc_aucs = []
    for c in range(5):
        y_c = (y_test == c).astype(int)
        score_c = probs[:, c]
        fpr, tpr, _ = roc_curve(y_c, score_c)
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2.0, color=colors[c], label=f"{CLASS_LABELS[c]} (AUC = {roc_auc:.3f})")

    macro_auc = float(np.mean(roc_aucs))
    plt.plot([0, 1], [0, 1], color="grey", lw=1.2, ls="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate", fontweight="bold")
    plt.title(f"One-vs-Rest ROC Curves (Macro-Average AUC = {macro_auc:.3f})", fontweight="bold")
    plt.grid(True, alpha=0.3, ls="--")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    roc_fig_path = figures_dir / "fig_roc_curves_5class.png"
    plt.savefig(roc_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved ROC curves to: {roc_fig_path}")

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS AND FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
