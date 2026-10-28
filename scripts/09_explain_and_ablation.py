"""Phase 2: SHAP Explainability & Leave-One-Group-Out Feature Ablation Study.

Generates:
- Figure: Global SHAP Feature Importance Bar Plot (Top 20 Features, 300 DPI, Fig 5)
- Figure: Leave-One-Group-Out Feature Ablation Study Bar Chart (300 DPI, Fig 6)
- Metrics Table: Feature ablation accuracy drops for the 5 feature categories
"""

import sys
import time
from pathlib import Path
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import accuracy_score
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.features.time_series import FEATURE_NAMES_63, FEATURE_GROUPS_63


CATEGORY_COLORS = {
    "statistical": "#1f77b4",        # Blue
    "burst_instability": "#d62728",   # Red
    "spectral_temporal": "#2ca02c",   # Green
    "basis_dependent": "#9467bd",     # Purple
    "channel_interaction": "#ff7f0e", # Orange
}

FEATURE_TO_GROUP = {}
for grp, f_list in FEATURE_GROUPS_63.items():
    for f in f_list:
        FEATURE_TO_GROUP[f] = grp


def load_config() -> dict:
    with open(ROOT_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    data_dir = ROOT_DIR / "data" / "processed"
    train_path = data_dir / "train_multiattack.csv"
    test_path = data_dir / "test_multiattack.csv"
    ckpt_path = data_dir / "checkpoints" / "xgb_multiattack.joblib"

    metrics_dir = ROOT_DIR / "results" / "metrics"
    figures_dir = ROOT_DIR / "results" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 2: SHAP EXPLAINABILITY & FEATURE GROUP ABLATION")
    print("=" * 70)

    print("\nLoading datasets and model checkpoint...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    model = joblib.load(ckpt_path)

    X_train = train_df[FEATURE_NAMES_63].values
    y_train = train_df["block_class"].values.astype(int)

    X_test = test_df[FEATURE_NAMES_63].values
    y_test = test_df["block_class"].values.astype(int)

    # -------------------------------------------------------------
    # 1. SHAP TreeExplainer & Global Feature Importance (Figure 5)
    # -------------------------------------------------------------
    print("\n[1/2] Computing SHAP TreeExplainer Values...")
    explainer = shap.TreeExplainer(model)

    # Sample 1,500 test points for fast exact Shapley values
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=min(1500, len(X_test)), replace=False)
    X_sample = X_test[sample_idx]

    shap_values = explainer.shap_values(X_sample)
    # For multi-class, shap_values is an array of shape (N, D, C) or list of (N, D)
    if isinstance(shap_values, list):
        # List of 5 arrays each of shape (N, D)
        global_imp = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)
    else:
        # Array of shape (N, D, C)
        global_imp = np.mean(np.mean(np.abs(shap_values), axis=0), axis=1)

    imp_df = pd.DataFrame({
        "feature": FEATURE_NAMES_63,
        "importance": global_imp,
        "category": [FEATURE_TO_GROUP.get(f, "other") for f in FEATURE_NAMES_63],
    }).sort_values("importance", ascending=True)

    imp_df.to_csv(metrics_dir / "multiattack_feature_importance_shap.csv", index=False)

    # Plot top 20 features
    top20 = imp_df.tail(20)
    plt.figure(figsize=(9, 7.5), dpi=300)
    bar_colors = [CATEGORY_COLORS.get(cat, "#333333") for cat in top20["category"]]
    plt.barh(top20["feature"], top20["importance"], color=bar_colors, alpha=0.85, edgecolor="black", lw=0.5)

    plt.xlabel("Mean |SHAP value| (Average Impact on Model Output)", fontweight="bold")
    plt.title("Global Physics-Informed Feature Importance (XGBoost, 5 Classes)", fontweight="bold", fontsize=12)
    plt.grid(True, axis="x", alpha=0.3, ls="--")

    # Add custom legend for categories
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[c], label=c.replace("_", " ").title())
        for c in ["channel_interaction", "basis_dependent", "burst_instability", "spectral_temporal", "statistical"]
    ]
    plt.legend(handles=handles, title="Physics Feature Category", loc="lower right", fontsize=8.5)
    plt.tight_layout()

    shap_fig_path = figures_dir / "fig_shap_multiattack.png"
    plt.savefig(shap_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP feature importance plot to: {shap_fig_path}")

    # -------------------------------------------------------------
    # 2. Leave-One-Group-Out Feature Ablation Study (Figure 6)
    # -------------------------------------------------------------
    print("\n[2/2] Running Leave-One-Group-Out Feature Ablation Study...")
    preds_full = model.predict(X_test)
    baseline_acc = float(accuracy_score(y_test, preds_full)) * 100.0
    print(f"  Baseline Accuracy with all 63 features: {baseline_acc:.2f}%")

    ablation_results = []
    # Train fast ablation models on 50k subsample of training data for speed
    sub_n = min(len(X_train), 50000)
    train_sub_idx = rng.choice(len(X_train), size=sub_n, replace=False)
    X_tr_sub = X_train[train_sub_idx]
    y_tr_sub = y_train[train_sub_idx]

    for grp_name, remove_features in FEATURE_GROUPS_63.items():
        print(f"  Ablating group: {grp_name} ({len(remove_features)} features removed)...")
        remain_idx = [i for i, f in enumerate(FEATURE_NAMES_63) if f not in remove_features]

        X_tr_abl = X_tr_sub[:, remain_idx]
        X_te_abl = X_test[:, remain_idx]

        m_abl = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            eval_metric="mlogloss",
            tree_method="hist",
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.80,
            random_state=42,
            n_jobs=-1,
        )
        m_abl.fit(X_tr_abl, y_tr_sub)
        p_abl = m_abl.predict(X_te_abl)
        abl_acc = float(accuracy_score(y_test, p_abl)) * 100.0
        acc_drop = max(0.0, baseline_acc - abl_acc)

        ablation_results.append({
            "group_name": grp_name,
            "display_name": grp_name.replace("_", " ").title(),
            "features_removed": len(remove_features),
            "ablated_accuracy": abl_acc,
            "accuracy_drop_pct": acc_drop,
        })
        print(f"    Accuracy: {abl_acc:.2f}% | Drop: -{acc_drop:.2f}%")

    abl_df = pd.DataFrame(ablation_results).sort_values("accuracy_drop_pct", ascending=True)
    abl_csv = metrics_dir / "multiattack_feature_ablation.csv"
    abl_df.to_csv(abl_csv, index=False)
    print(f"  Saved ablation metrics to: {abl_csv}")

    # Plot ablation bar chart (Figure 6)
    plt.figure(figsize=(8, 5), dpi=300)
    bars = plt.barh(abl_df["display_name"], abl_df["accuracy_drop_pct"], color="#d62728", alpha=0.85, edgecolor="black")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"+{w:.2f}% drop", va="center", fontweight="bold", fontsize=9)

    plt.xlabel("Accuracy Drop When Group Removed (%)", fontweight="bold")
    plt.title("Leave-One-Group-Out Feature Ablation Study (5 Classes)", fontweight="bold", fontsize=12)
    plt.grid(True, axis="x", alpha=0.3, ls="--")
    plt.xlim(0, max(abl_df["accuracy_drop_pct"]) + 2.0)
    plt.tight_layout()

    abl_fig_path = figures_dir / "fig_ablation_5class.png"
    plt.savefig(abl_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved ablation plot to: {abl_fig_path}")

    print("\n" + "=" * 70)
    print("SHAP & ABLATION STUDY FINISHED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
