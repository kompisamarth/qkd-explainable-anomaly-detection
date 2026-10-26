"""Publication-quality figure generation and XAI analysis script for BB84 QKD.

Generates 300 DPI figures for the IEEE EL conference paper:
- Figure 1: Detection Latency (Delta_T) vs. Eve Interception Rate (p_eve)
- Figure 2: ROC and Precision-Recall Curves on Fiber Test Set
- Figure 3: SHAP Global Feature Importance and Beeswarm Summary Plots
- Figure 4: Cross-Channel Noise Generalization Degradation (Fiber -> Free-Space)
- Figure 5: Chronological Single-Session Attack Onset Trace (Stealthy p_eve=0.20)
"""

import sys
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.features.time_series import FEATURE_NAMES, CausalTimeSeriesExtractor
from src.simulator.bb84_engine import BB84Simulator
from src.models.ml_classifiers import QKDMLDetector
from src.models.explainability import QKDExplainer


# Set publication aesthetics
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
})


def load_config() -> dict:
    with open(ROOT_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def plot_latency_comparison(metrics_dir: Path, fig_dir: Path) -> None:
    """Figure 1: Operational Detection Latency Delta_T vs. p_eve."""
    latency_file = metrics_dir / "detection_latency_comparison.csv"
    if not latency_file.exists():
        print("  Latency comparison file not found, skipping Fig 1.")
        return

    df = pd.read_csv(latency_file)
    plt.figure(figsize=(9, 5.5), dpi=300)

    model_colors = {
        "Static 11%": "#7f7f7f",
        "Hoeffding Single": "#ff7f0e",
        "Hoeffding Windowed": "#2ca02c",
        "Random Forest": "#1f77b4",
        "XGBoost": "#d62728",
    }
    model_markers = {
        "Static 11%": "s",
        "Hoeffding Single": "^",
        "Hoeffding Windowed": "v",
        "Random Forest": "o",
        "XGBoost": "D",
    }

    for model_name, grp in df.groupby("model"):
        grp_sorted = grp.sort_values(by="p_eve")
        plt.plot(
            grp_sorted["p_eve"] * 100,
            grp_sorted["mean_delta_t"],
            label=model_name,
            color=model_colors.get(model_name, None),
            marker=model_markers.get(model_name, "o"),
            alpha=0.9,
        )

    # Shade the stealthy sub-11% QBER regime (p_eve in 10% - 30%)
    plt.axvspan(10, 30, color="#fff2cc", alpha=0.5, label="Stealth Regime (Sub-11% QBER)")
    plt.axvline(30, color="#d9534f", linestyle=":", alpha=0.8)

    plt.xlabel(r"Eve Interception Rate $p_{\text{eve}}$ (%)", fontweight="bold")
    plt.ylabel(r"Detection Latency $\Delta T$ (Elapsed Blocks)", fontweight="bold")
    plt.title(r"Operational Detection Latency $\Delta T$ vs. Adversary Interception Rate", fontweight="bold", pad=12)
    plt.yscale("log")
    plt.ylim(1, 120)
    plt.xlim(4, 101)
    plt.grid(True, which="both")
    plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    plt.tight_layout()

    out_path = fig_dir / "fig1_latency_vs_interception.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 1 to {out_path.name}")


def plot_roc_pr_curves(test_fiber_df: pd.DataFrame, models: dict, fig_dir: Path) -> None:
    """Figure 2: ROC and Precision-Recall Curves on Fiber Test Set."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    y_true = test_fiber_df["is_attack_active"].values

    colors = {
        "Static 11% Threshold": "#7f7f7f",
        "Hoeffding (Windowed W=10)": "#2ca02c",
        "Random Forest (Calibrated)": "#1f77b4",
        "XGBoost (Calibrated)": "#d62728",
    }

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(test_fiber_df)[:, 1]
        else:
            continue

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(rec, prec)

        c = colors.get(name, "#333333")
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=c, lw=2.0)
        axes[1].plot(rec, prec, label=f"{name} (AUPRC = {pr_auc:.3f})", color=c, lw=2.0)

    # ROC Subplot
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    axes[0].axvline(0.01, color="red", linestyle=":", alpha=0.7, label="Operational FPR Target <= 1%")
    axes[0].set_xlabel("False Positive Rate (FPR)", fontweight="bold")
    axes[0].set_ylabel("True Positive Rate (TPR / Recall)", fontweight="bold")
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
    axes[0].set_xlim(-0.01, 0.25)  # Zoom in on operationally meaningful low-FPR region
    axes[0].set_ylim(0.40, 1.02)
    axes[0].grid(True)
    axes[0].legend(loc="lower right", frameon=True)

    # PR Subplot
    axes[1].set_xlabel("Recall", fontweight="bold")
    axes[1].set_ylabel("Precision", fontweight="bold")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].set_xlim(0.40, 1.02)
    axes[1].set_ylim(0.40, 1.02)
    axes[1].grid(True)
    axes[1].legend(loc="lower left", frameon=True)

    plt.tight_layout()
    out_path = fig_dir / "fig2_roc_pr_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 2 to {out_path.name}")


def plot_generalization_degradation(metrics_dir: Path, fig_dir: Path) -> None:
    """Figure 4: Cross-Channel Noise Generalization Degradation."""
    cross_file = metrics_dir / "cross_channel_generalization.csv"
    if not cross_file.exists():
        print("  Cross channel file not found, skipping Fig 4.")
        return

    df = pd.read_csv(cross_file)
    # Filter to main models
    plot_models = ["Hoeffding (Windowed W=10)", "Random Forest (Calibrated)", "XGBoost (Calibrated)"]
    df_sub = df[df["model"].isin(plot_models)].copy()

    x = np.arange(len(df_sub))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Subplot A: F1-Score in Fiber vs. Free-Space
    axes[0].bar(x - width/2, df_sub["fiber_f1"], width, label="Fiber (Stationary Depolarizing)", color="#1f77b4", edgecolor="black")
    axes[0].bar(x + width/2, df_sub["fso_f1"], width, label="Free-Space (Atmospheric Scintillation)", color="#ff7f0e", edgecolor="black")
    axes[0].set_ylabel("F1-Score", fontweight="bold")
    axes[0].set_title("Cross-Channel F1-Score Performance", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace(" (Calibrated)", "") for m in df_sub["model"]], rotation=10, ha="right")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis="y")
    axes[0].legend(loc="lower right")

    # Subplot B: False Positive Rate (FPR) shift
    axes[1].bar(x - width/2, df_sub["fiber_fpr"] * 100, width, label="Fiber FPR (%)", color="#2ca02c", edgecolor="black")
    axes[1].bar(x + width/2, df_sub["fso_fpr"] * 100, width, label="Free-Space FPR (%)", color="#d62728", edgecolor="black")
    axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.8, label="Operational Limit (1%)")
    axes[1].set_ylabel("False Positive Rate (%)", fontweight="bold")
    axes[1].set_title("False Alarm Shift Under Atmospheric Turbulence", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.replace(" (Calibrated)", "") for m in df_sub["model"]], rotation=10, ha="right")
    axes[1].grid(axis="y")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    out_path = fig_dir / "fig4_cross_channel_generalization.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 4 to {out_path.name}")


def plot_session_trace(
    rf_detector: QKDMLDetector,
    fig_dir: Path,
    cfg: dict
) -> None:
    """Figure 5: Chronological Single-Session Trace showing attack onset at block 101."""
    sim = BB84Simulator(raw_photons_per_block=1024, base_qber=0.03, random_state=123)
    extractor = CausalTimeSeriesExtractor(window_size=10, ema_span=5)

    # Simulate stealthy fractional attack: p_eve = 0.20
    seq = sim.simulate_session(
        session_id=777,
        target_attack_class=2,
        channel_type="fiber",
        blocks_per_session=150,
        attack_start_block=75,
        p_eve_fractional=0.20,
    )

    df_feat = extractor.extract_from_session(
        session_id=777,
        qbers=seq.block_qbers,
        sifted_lens=seq.block_sifted_lens,
        block_classes=seq.block_classes,
        attack_class=seq.attack_class,
        channel_type="fiber",
        p_eve=0.20,
        attack_start_block=75,
    )

    # Model predictions
    probs = rf_detector.predict_proba(df_feat)[:, 1]
    blocks = df_feat["block_index"].values
    qbers = df_feat["qber_current"].values
    rolling_means = df_feat["rolling_mean"].values

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, dpi=300)

    # Top Subplot: QBER & Static vs. Dynamic Thresholds
    axes[0].plot(blocks, qbers * 100, "o-", color="#8c564b", alpha=0.5, label="Raw Block QBER ($e_t$)", markersize=4)
    axes[0].plot(blocks, rolling_means * 100, color="#1f77b4", lw=2.5, label="Rolling Mean ($W=10$)")
    axes[0].axhline(11.0, color="#d62728", linestyle="--", lw=2, label="Static Abort Threshold (11%)")
    axes[0].axvline(75, color="#2ca02c", linestyle="-", lw=2.5, label=r"Attack Onset $T_{\text{attack}} = 75$ ($p_{\text{eve}}=20\%$)")
    axes[0].axvspan(75, 150, color="#e8f5e9", alpha=0.6)
    axes[0].set_ylabel("QBER (%)", fontweight="bold")
    axes[0].set_title(r"Temporal Error Evolution During Stealthy Eavesdropping ($p_{\text{eve}}=0.20$)", fontweight="bold")
    axes[0].set_ylim(0, 14)
    axes[0].grid(True)
    axes[0].legend(loc="upper left", frameon=True)

    # Bottom Subplot: ML Alarm Probability
    axes[1].plot(blocks, probs, color="#d62728", lw=2.2, label=r"ML Posterior $P(\text{Attack} \mid X_t)$")
    axes[1].axhline(rf_detector.calibrated_threshold, color="black", linestyle=":", label=f"Decision Threshold ({rf_detector.calibrated_threshold:.2f})")
    axes[1].axvline(75, color="#2ca02c", linestyle="-", lw=2.5)
    axes[1].axvspan(75, 150, color="#e8f5e9", alpha=0.6)
    
    # Identify detection point
    alarm_mask = (blocks >= 75) & (probs >= rf_detector.calibrated_threshold)
    if np.any(alarm_mask):
        first_alarm = blocks[alarm_mask][0]
        axes[1].scatter([first_alarm], [probs[blocks == first_alarm][0]], color="gold", edgecolor="black", s=140, zorder=5, label=f"Alarm Flagged ($t={first_alarm}$, $\\Delta T = {first_alarm-75}$)")
        axes[0].axvline(first_alarm, color="gold", linestyle=":", lw=2)

    axes[1].set_xlabel("Elapsed Transmission Block Index", fontweight="bold")
    axes[1].set_ylabel(r"$P(\text{Attack})$", fontweight="bold")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True)
    axes[1].legend(loc="upper left", frameon=True)

    plt.tight_layout()
    out_path = fig_dir / "fig5_session_temporal_trace.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 5 to {out_path.name}")


class StaticThresholdDetector:
    """Standard static QBER abort threshold (11%)."""
    def __init__(self, threshold: float = 0.11):
        self.threshold = threshold

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        qber = X["qber_current"].values if "qber_current" in X.columns else X.iloc[:, 0].values
        return (qber >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        qber = X["qber_current"].values if "qber_current" in X.columns else X.iloc[:, 0].values
        p_attack = np.clip(qber / self.threshold, 0.0, 1.0)
        return np.column_stack([1.0 - p_attack, p_attack])


def main():
    print("=" * 70)
    print("BB84 EXPLAINABLE AI (XAI) & PUBLICATION FIGURE GENERATION")
    print("=" * 70)

    cfg = load_config()
    proc_dir = ROOT_DIR / cfg["paths"]["processed_data"]
    metrics_dir = ROOT_DIR / cfg["paths"]["metrics"]
    fig_dir = ROOT_DIR / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data & Trained Models
    print("\n[1/4] Loading Validation/Test Datasets & Checkpoints...")
    test_fiber_df = pd.read_csv(proc_dir / "test_fiber.csv")
    models_dir = proc_dir / "checkpoints"

    rf_binary = QKDMLDetector.load(models_dir / "rf_binary.joblib")
    xgb_binary = QKDMLDetector.load(models_dir / "xgb_binary.joblib")

    from src.models.hoeffding_baseline import HoeffdingDetector
    hoeffding_win = HoeffdingDetector(base_qber=0.03, alpha=0.01, mode="windowed", window_size=10)
    static_det = StaticThresholdDetector(threshold=0.11)

    eval_models = {
        "Static 11% Threshold": static_det,
        "Hoeffding (Windowed W=10)": hoeffding_win,
        "Random Forest (Calibrated)": rf_binary,
        "XGBoost (Calibrated)": xgb_binary,
    }

    # 2. Generate SHAP Explanations
    print("\n[2/4] Running SHAP TreeExplainer & Feature Attribution...")
    explainer = QKDExplainer(rf_binary, feature_names=FEATURE_NAMES)
    shap_vals, X_eval = explainer.compute_shap_values(test_fiber_df, sample_size=1200, random_state=42)

    importance_df = explainer.get_feature_importance_df(shap_vals, X_eval)
    importance_df.to_csv(metrics_dir / "feature_importance_shap.csv", index=False)
    print(f"  Top 5 Features by Mean |SHAP|:\n{importance_df.head(5).to_string(index=False)}")

    # Plot SHAP Figures
    print("\n[3/4] Plotting SHAP Visualizations...")
    explainer.plot_bar_importance(
        importance_df,
        save_path=fig_dir / "fig3b_shap_importance_bar.png",
        max_display=10,
        title="Global Feature Importance: Top Temporal Predictors",
    )
    print(f"  Saved Figure 3b to fig3b_shap_importance_bar.png")

    explainer.plot_summary(
        shap_vals,
        X_eval,
        save_path=fig_dir / "fig3_shap_beeswarm.png",
        max_display=10,
        title="SHAP Beeswarm: Error Drift & Variance Impact on Detection",
    )
    print(f"  Saved Figure 3 to fig3_shap_beeswarm.png")

    # 3. Generate Benchmark Publication Figures
    print("\n[4/4] Generating Publication Benchmark Figures...")
    plot_latency_comparison(metrics_dir, fig_dir)
    plot_roc_pr_curves(test_fiber_df, eval_models, fig_dir)
    plot_generalization_degradation(metrics_dir, fig_dir)
    plot_session_trace(rf_binary, fig_dir, cfg)

    print("\n" + "=" * 70)
    print("ALL PUBLICATION FIGURES GENERATED AT 300 DPI!")
    print("=" * 70)


if __name__ == "__main__":
    main()
