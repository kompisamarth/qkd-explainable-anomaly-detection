"""Hoeffding's Inequality Statistical Baseline for BB84 Eavesdropping Detection.

Implements the dynamic threshold framework proposed by Lee, Sohn, & Lee (2022):
  e_thresh = e_0 + sqrt(ln(1 / alpha) / (2 * N))
providing a mathematically sound FPR <= alpha statistical test.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd


class HoeffdingDetector:
    """Statistical detector based on Hoeffding's Inequality.
    
    References:
        Lee, S., Sohn, I., & Lee, W. (2022). Dynamic threshold for eavesdropping
        detection in quantum key distribution. IEEE Access.
    """

    def __init__(
        self,
        base_qber: float = 0.03,
        alpha: float = 0.01,
        mode: str = "single_block",  # 'single_block' or 'windowed'
        window_size: int = 10,
    ):
        self.base_qber = float(base_qber)
        self.alpha = float(alpha)
        self.mode = mode
        self.window_size = int(window_size)
        self.log_inv_alpha = np.log(1.0 / self.alpha)

    def compute_threshold(self, sifted_bits: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculates dynamic Hoeffding threshold: e_0 + sqrt(ln(1/alpha) / (2 * N))."""
        effective_n = sifted_bits * (self.window_size if self.mode == "windowed" else 1.0)
        margin = np.sqrt(self.log_inv_alpha / (2.0 * np.maximum(effective_n, 1.0)))
        return self.base_qber + margin

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Flags an alarm (1) if observed QBER exceeds dynamic Hoeffding threshold, else (0)."""
        if isinstance(X, pd.DataFrame):
            if self.mode == "windowed" and "rolling_mean" in X.columns:
                qber_vals = X["rolling_mean"].values
            else:
                qber_vals = X["qber_current"].values
            sifted = X["sifted_bits"].values if "sifted_bits" in X.columns else np.full_like(qber_vals, 512.0)
        else:
            # Assume column 0 is qber_current, column 1 is rolling_mean, column 13 is sifted_bits
            if self.mode == "windowed" and X.shape[1] > 1:
                qber_vals = X[:, 1]
            else:
                qber_vals = X[:, 0]
            sifted = X[:, 13] if X.shape[1] > 13 else np.full_like(qber_vals, 512.0)

        thresholds = self.compute_threshold(sifted)
        return (qber_vals > thresholds).astype(int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Returns pseudo-probabilities based on distance to Hoeffding boundary."""
        if isinstance(X, pd.DataFrame):
            if self.mode == "windowed" and "rolling_mean" in X.columns:
                qber_vals = X["rolling_mean"].values
            else:
                qber_vals = X["qber_current"].values
            sifted = X["sifted_bits"].values if "sifted_bits" in X.columns else np.full_like(qber_vals, 512.0)
        else:
            if self.mode == "windowed" and X.shape[1] > 1:
                qber_vals = X[:, 1]
            else:
                qber_vals = X[:, 0]
            sifted = X[:, 13] if X.shape[1] > 13 else np.full_like(qber_vals, 512.0)

        thresholds = self.compute_threshold(sifted)
        # Scale margin: when qber == threshold, prob = 0.5
        effective_n = sifted * (self.window_size if self.mode == "windowed" else 1.0)
        margin = np.sqrt(self.log_inv_alpha / (2.0 * np.maximum(effective_n, 1.0)))
        z_score = (qber_vals - thresholds) / (margin + 1e-9)
        # Map through logistic function
        p_attack = 1.0 / (1.0 + np.exp(-3.0 * z_score))
        p_benign = 1.0 - p_attack
        return np.column_stack([p_benign, p_attack])
