"""Explainable AI (XAI) module using SHAP TreeExplainer for BB84 QKD Anomaly Detection.

Produces:
- Global feature importance rankings
- SHAP Beeswarm summary distributions
- Feature-level attributions unmasking stealthy Fractional Intercept-Resend attacks
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


class QKDExplainer:
    """SHAP TreeExplainer wrapper for tree-based QKD anomaly detection models."""

    def __init__(self, detector_model: Any, feature_names: List[str]):
        self.detector_model = detector_model
        self.feature_names = feature_names
        
        # Instantiate TreeExplainer
        raw_model = getattr(detector_model, "model", detector_model)
        self.explainer = shap.TreeExplainer(raw_model)

    def compute_shap_values(
        self,
        X: pd.DataFrame,
        sample_size: Optional[int] = 1000,
        random_state: int = 42
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Calculates SHAP values over a sampled subset of the input data."""
        X_sub = X[self.feature_names]
        if sample_size is not None and len(X_sub) > sample_size:
            X_eval = X_sub.sample(n=sample_size, random_state=random_state)
        else:
            X_eval = X_sub

        shap_vals = self.explainer.shap_values(X_eval)
        
        # In binary classification, some shap versions return a list of [class_0, class_1]
        # or an array of shape (N, D, 2)
        if isinstance(shap_vals, list):
            # Binary classification: take class 1 (Attack)
            shap_eval = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            shap_eval = shap_vals[:, :, 1]
        else:
            shap_eval = shap_vals

        return shap_eval, X_eval

    def get_feature_importance_df(
        self,
        shap_values: np.ndarray,
        X_eval: pd.DataFrame
    ) -> pd.DataFrame:
        """Returns a ranked DataFrame of mean absolute SHAP values per feature."""
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        df_importance = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
        return df_importance

    def plot_summary(
        self,
        shap_values: np.ndarray,
        X_eval: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = None,
        max_display: int = 12,
        title: str = "SHAP Summary: Temporal Feature Attributions",
    ) -> None:
        """Renders and saves a publication-quality SHAP beeswarm plot."""
        plt.figure(figsize=(10, 6), dpi=300)
        shap.summary_plot(
            shap_values,
            X_eval,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
            plot_size=(10, 6)
        )
        plt.title(title, fontsize=13, fontweight="bold", pad=14)
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_bar_importance(
        self,
        importance_df: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = None,
        max_display: int = 10,
        title: str = "Global Feature Importance (Mean |SHAP Value|)",
    ) -> None:
        """Renders and saves a horizontal bar chart of top feature importances."""
        df_top = importance_df.head(max_display).iloc[::-1]  # Highest on top
        
        plt.figure(figsize=(9, 5), dpi=300)
        bars = plt.barh(df_top["feature"], df_top["mean_abs_shap"], color="#1f77b4", edgecolor="black", alpha=0.85)
        plt.xlabel("Mean |SHAP Value| (Impact on Attack Classification)", fontsize=11, fontweight="bold")
        plt.title(title, fontsize=12, fontweight="bold", pad=12)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                ha="left",
                va="center",
                fontsize=9
            )
            
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
