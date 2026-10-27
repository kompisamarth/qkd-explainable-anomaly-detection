"""Causal Time-Series Feature Engineering for BB84 QBER Error Sequences.

Computes backward-looking sliding window features with ZERO future data leakage.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


FEATURE_NAMES = [
    "qber_current",            # 1. Instantaneous block QBER e_t
    "rolling_mean",            # 2. Rolling mean mu_t over window W
    "rolling_std",             # 3. Rolling standard deviation sigma_t
    "rolling_var",             # 4. Rolling variance sigma_t^2
    "fano_factor",             # 5. Variance-to-Mean Ratio (sigma_t^2 / mu_t)
    "linear_drift_slope",      # 6. First-order drift / linear regression slope across W
    "ema_divergence",          # 7. Divergence from exponential moving average (e_t - EMA_t)
    "rolling_skewness",        # 8. Fisher-Pearson coefficient of skewness
    "rolling_kurtosis",        # 9. Excess kurtosis of QBER distribution in window
    "shannon_entropy",         # 10. Shannon entropy of binned error distribution
    "burstiness_index",        # 11. Burstiness (sigma - mu) / (sigma + mu)
    "lag1_autocorr",           # 12. Empirical lag-1 autocorrelation across window
    "qber_range",              # 13. Peak-to-peak spread (max - min) in window
    "sifted_bits",             # 14. Instantaneous sifted block length N_t
]


class CausalTimeSeriesExtractor:
    """Extracts causal temporal features over a backward sliding window of length W.
    
    Guarantees that at timestep t, only blocks tau <= t are accessed, ensuring
    strict operational validity for sequential anomaly detection and latency benchmarking.
    """

    def __init__(self, window_size: int = 10, ema_span: int = 5):
        self.window_size = int(window_size)
        self.ema_span = int(ema_span)
        self.alpha_ema = 2.0 / (self.ema_span + 1.0)
        self.feature_names = FEATURE_NAMES

    def extract_single_window(
        self,
        qber_window: np.ndarray,
        current_sifted_bits: int,
        current_ema: float
    ) -> np.ndarray:
        """Extracts the 14-feature vector from a single chronological window of length W.
        
        Args:
            qber_window: 1D array of shape (W,) containing [e_{t-W+1}, ..., e_t].
            current_sifted_bits: Sifted bit count N_t of the current block.
            current_ema: Exponential moving average computed up to timestep t.
            
        Returns:
            1D numpy array of shape (14,) with extracted feature values.
        """
        w = len(qber_window)
        e_t = float(qber_window[-1])
        
        mean_val = float(np.mean(qber_window))
        std_val = float(np.std(qber_window, ddof=1)) if w > 1 else 0.0
        var_val = float(std_val ** 2)
        
        # Fano factor (Variance / Mean)
        fano = var_val / (mean_val + 1e-9)
        
        # First-order drift slope (linear regression slope over [0, ..., W-1])
        x = np.arange(w, dtype=np.float64)
        x_centered = x - np.mean(x)
        slope = float(np.dot(x_centered, qber_window) / (np.dot(x_centered, x_centered) + 1e-12))
        
        # EMA Divergence
        ema_div = e_t - current_ema
        
        # Skewness and Kurtosis
        if std_val > 1e-8 and w >= 4:
            skew_val = float(stats.skew(qber_window, bias=False))
            kurt_val = float(stats.kurtosis(qber_window, bias=False))
        else:
            skew_val = 0.0
            kurt_val = 0.0
            
        # Shannon Entropy across 5 discrete bins in [0, 0.35]
        hist, _ = np.histogram(qber_window, bins=5, range=(0.0, 0.35))
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]
        entropy_val = float(-np.sum(probs * np.log2(probs)))
        
        # Burstiness Index: (std - mean) / (std + mean) bounded in [-1, 1]
        sum_ms = std_val + mean_val
        burstiness = float((std_val - mean_val) / sum_ms) if sum_ms > 1e-8 else 0.0
        burstiness = np.clip(burstiness, -1.0, 1.0)
        
        # Lag-1 Autocorrelation across the window
        if std_val > 1e-8 and w > 2:
            denom = np.sum((qber_window - mean_val) ** 2)
            if denom > 1e-12:
                num = np.sum((qber_window[:-1] - mean_val) * (qber_window[1:] - mean_val))
                lag1 = float(num / denom)
            else:
                lag1 = 0.0
        else:
            lag1 = 0.0
            
        # Range
        qber_range = float(np.max(qber_window) - np.min(qber_window))
        
        return np.array([
            e_t,
            mean_val,
            std_val,
            var_val,
            fano,
            slope,
            ema_div,
            skew_val,
            kurt_val,
            entropy_val,
            burstiness,
            lag1,
            qber_range,
            float(current_sifted_bits),
        ], dtype=np.float64)

    def extract_from_session(
        self,
        session_id: int,
        qbers: np.ndarray,
        sifted_lens: np.ndarray,
        block_classes: np.ndarray,
        attack_class: int,
        channel_type: str,
        p_eve: float,
        attack_start_block: int = 101,
    ) -> pd.DataFrame:
        """Causally extracts feature rows from an entire chronological session.

        Returns a DataFrame where each row corresponds to block timestep t (for t >= W).
        Vectorized over all windows; numerically equivalent to calling
        extract_single_window() on each window (see extract_from_session_reference).
        """
        qbers = np.asarray(qbers, dtype=np.float64)
        num_blocks = len(qbers)
        w = self.window_size
        if num_blocks < w:
            return pd.DataFrame()

        # Causal recursive EMA (inherently sequential, but only O(B))
        ema = np.empty(num_blocks, dtype=np.float64)
        ema[0] = qbers[0]
        for i in range(1, num_blocks):
            ema[i] = self.alpha_ema * qbers[i] + (1.0 - self.alpha_ema) * ema[i - 1]

        # Row r holds window [e_{t-W+1}, ..., e_t] for t = r + W (1-indexed)
        win = np.lib.stride_tricks.sliding_window_view(qbers, w)
        end_idx = np.arange(w - 1, num_blocks)
        e_t = win[:, -1]

        mean_val = win.mean(axis=1)
        std_val = win.std(axis=1, ddof=1) if w > 1 else np.zeros(len(win))
        var_val = std_val ** 2
        fano = var_val / (mean_val + 1e-9)

        x_centered = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
        slope = (win @ x_centered) / (np.dot(x_centered, x_centered) + 1e-12)

        ema_div = e_t - ema[end_idx]

        has_spread = std_val > 1e-8
        if w >= 4:
            with np.errstate(all="ignore"):
                skew_val = np.where(has_spread, stats.skew(win, axis=1, bias=False), 0.0)
                kurt_val = np.where(has_spread, stats.kurtosis(win, axis=1, bias=False), 0.0)
        else:
            skew_val = np.zeros(len(win))
            kurt_val = np.zeros(len(win))

        # Shannon entropy over 5 bins in [0, 0.35], same edges/semantics as np.histogram
        edges = np.linspace(0.0, 0.35, 6)
        bin_idx = np.searchsorted(edges, win, side="right") - 1
        bin_idx[win == edges[-1]] = 4          # right edge is inclusive
        in_range = (win >= edges[0]) & (win <= edges[-1])
        counts = np.stack([np.sum((bin_idx == k) & in_range, axis=1) for k in range(5)], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            probs = counts / counts.sum(axis=1, keepdims=True)
            plogp = np.where(probs > 0, probs * np.log2(np.where(probs > 0, probs, 1.0)), 0.0)
        entropy_val = -plogp.sum(axis=1)

        sum_ms = std_val + mean_val
        with np.errstate(divide="ignore", invalid="ignore"):
            burstiness = np.where(sum_ms > 1e-8, (std_val - mean_val) / sum_ms, 0.0)
        burstiness = np.clip(burstiness, -1.0, 1.0)

        centered = win - mean_val[:, None]
        denom = np.sum(centered ** 2, axis=1)
        num = np.sum(centered[:, :-1] * centered[:, 1:], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            lag1 = np.where(has_spread & (w > 2) & (denom > 1e-12), num / denom, 0.0)

        qber_range = win.max(axis=1) - win.min(axis=1)
        sifted = np.asarray(sifted_lens, dtype=np.float64)[end_idx]

        t_index = end_idx + 1
        block_cls = np.asarray(block_classes)[end_idx].astype(int)
        df = pd.DataFrame({
            "session_id": session_id,
            "block_index": t_index,
            "channel_type": channel_type,
            "session_attack_class": attack_class,
            "session_p_eve": p_eve,
            "attack_start_block": attack_start_block,
            "is_attack_active": ((t_index >= attack_start_block) & (attack_class > 0)).astype(int),
            "block_class": block_cls,
        })
        feat_cols = [e_t, mean_val, std_val, var_val, fano, slope, ema_div, skew_val,
                     kurt_val, entropy_val, burstiness, lag1, qber_range, sifted]
        for f_name, col in zip(self.feature_names, feat_cols):
            df[f_name] = np.asarray(col, dtype=np.float64)
        return df

    def extract_from_session_reference(
        self,
        session_id: int,
        qbers: np.ndarray,
        sifted_lens: np.ndarray,
        block_classes: np.ndarray,
        attack_class: int,
        channel_type: str,
        p_eve: float,
        attack_start_block: int = 101,
    ) -> pd.DataFrame:
        """Original per-window loop implementation, kept as a correctness oracle for tests."""
        num_blocks = len(qbers)
        records = []
        
        # Initialize causal recursive EMA
        ema = float(qbers[0])
        
        for t in range(1, num_blocks + 1):
            idx = t - 1
            e_t = qbers[idx]
            
            # Update EMA recursively
            if t == 1:
                ema = float(e_t)
            else:
                ema = self.alpha_ema * float(e_t) + (1.0 - self.alpha_ema) * ema
                
            # Only generate feature vector once we have at least W blocks
            if t >= self.window_size:
                window = qbers[t - self.window_size : t]
                feat_vec = self.extract_single_window(
                    window,
                    current_sifted_bits=int(sifted_lens[idx]),
                    current_ema=ema,
                )
                
                row_dict = {
                    "session_id": session_id,
                    "block_index": t,
                    "channel_type": channel_type,
                    "session_attack_class": attack_class,
                    "session_p_eve": p_eve,
                    "attack_start_block": attack_start_block,
                    "is_attack_active": int(t >= attack_start_block and attack_class > 0),
                    "block_class": int(block_classes[idx]),
                }
                
                for f_name, f_val in zip(self.feature_names, feat_vec):
                    row_dict[f_name] = f_val
                    
                records.append(row_dict)
                
        return pd.DataFrame(records)
