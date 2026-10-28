"""Causal Time-Series Feature Engineering for BB84 QBER Error Sequences.

Computes backward-looking sliding window features with ZERO future data leakage.
"""

import warnings
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



# ==============================================================================
# PHASE 2: 63 PHYSICS-INFORMED FEATURES ACROSS 5 COMPLEMENTARY CATEGORIES
# ==============================================================================

FEATURE_GROUPS_63 = {
    "statistical": [
        "qber_mean", "qber_variance", "qber_std", "qber_skewness",
        "qber_kurtosis", "qber_iqr", "qber_cv"
    ],
    "burst_instability": [
        "spike_density", "burst_duration", "jump_energy", "temporal_drift_slope",
        "max_jump", "jump_count", "error_gap_mean", "error_gap_std",
        "error_gap_cv", "burstiness_index"
    ],
    "spectral_temporal": [
        "autocorr_lag_1", "autocorr_lag_2", "autocorr_lag_3", "autocorr_lag_4",
        "autocorr_lag_5", "dft_low_energy", "dft_mid_energy", "dft_high_energy",
        "spectral_entropy", "spectral_centroid", "shannon_entropy", "ema_divergence"
    ],
    "basis_dependent": [
        "qber_rectilinear", "qber_diagonal", "basis_asymmetry", "basis_ratio",
        "basis_burst_energy_z", "basis_burst_energy_x", "basis_variance_ratio",
        "basis_cross_correlation", "basis_skewness_diff", "basis_kurtosis_diff",
        "sift_ratio_z", "sift_ratio_x"
    ],
    "channel_interaction": [
        "key_generation_rate", "transmission_efficiency", "qber_loss_interaction",
        "qber_loss_product", "loss_anomaly_index", "dark_count_ratio",
        "multiscale_mean_w10", "multiscale_mean_w25", "multiscale_mean_w50",
        "multiscale_std_w10", "multiscale_std_w25", "multiscale_std_w50",
        "multiscale_fano_w10", "multiscale_fano_w25", "multiscale_fano_w50",
        "multiscale_drift_w10", "multiscale_drift_w25", "multiscale_drift_w50",
        "error_clustering_coefficient", "loss_gradient", "efficiency_entropy",
        "qber_stability_index"
    ],
}

FEATURE_NAMES_63 = []
for grp_key in ["statistical", "burst_instability", "spectral_temporal", "basis_dependent", "channel_interaction"]:
    FEATURE_NAMES_63.extend(FEATURE_GROUPS_63[grp_key])

assert len(FEATURE_NAMES_63) == 63, f"Expected exactly 63 features, got {len(FEATURE_NAMES_63)}"


class CausalMultiAttackFeatureExtractor:
    """Vectorized causal extractor for the 63 physics-informed features (W=50)."""

    def __init__(self, window_size: int = 50, ema_span: int = 10, base_qber: float = 0.03):
        self.window_size = int(window_size)
        self.ema_span = int(ema_span)
        self.alpha_ema = 2.0 / (self.ema_span + 1.0)
        self.base_qber = float(base_qber)
        self.feature_names = FEATURE_NAMES_63
        self.feature_groups = FEATURE_GROUPS_63

    def extract_from_session(self, seq) -> pd.DataFrame:
        """Extracts 63 features causally over sliding windows of length W."""
        qbers = np.asarray(seq.block_qbers, dtype=np.float64)
        num_blocks = len(qbers)
        w = self.window_size
        if num_blocks < w:
            return pd.DataFrame()

        # Arrays from session sequence
        q_z = np.asarray(seq.block_qbers_z if seq.block_qbers_z is not None else qbers, dtype=np.float64)
        q_x = np.asarray(seq.block_qbers_x if seq.block_qbers_x is not None else qbers, dtype=np.float64)
        sift_len = np.asarray(seq.block_sifted_lens, dtype=np.float64)
        sift_z = np.asarray(seq.block_sifted_z if seq.block_sifted_z is not None else (sift_len / 2), dtype=np.float64)
        sift_x = np.asarray(seq.block_sifted_x if seq.block_sifted_x is not None else (sift_len / 2), dtype=np.float64)
        raw_photons = np.asarray(seq.block_raw_photons if seq.block_raw_photons is not None else np.full(num_blocks, 1024), dtype=np.float64)
        det_eff = np.asarray(seq.block_transmission_effs if seq.block_transmission_effs is not None else np.full(num_blocks, 0.50), dtype=np.float64)
        gap_mean = np.asarray(seq.block_error_gap_means if seq.block_error_gap_means is not None else np.full(num_blocks, 33.0), dtype=np.float64)
        gap_std = np.asarray(seq.block_error_gap_stds if seq.block_error_gap_stds is not None else np.full(num_blocks, 15.0), dtype=np.float64)
        gap_cv = np.asarray(seq.block_error_gap_cvs if seq.block_error_gap_cvs is not None else np.full(num_blocks, 0.90), dtype=np.float64)
        clustering = np.asarray(seq.block_error_clusterings if seq.block_error_clusterings is not None else np.full(num_blocks, 0.03), dtype=np.float64)
        block_classes = np.asarray(seq.block_classes, dtype=np.int32)

        # Causal recursive EMA on QBER
        ema = np.empty(num_blocks, dtype=np.float64)
        ema[0] = qbers[0]
        for i in range(1, num_blocks):
            ema[i] = self.alpha_ema * qbers[i] + (1.0 - self.alpha_ema) * ema[i - 1]

        # Sliding window views (shape: (K, W) where K = num_blocks - W + 1)
        win_q = np.lib.stride_tricks.sliding_window_view(qbers, w)
        win_qz = np.lib.stride_tricks.sliding_window_view(q_z, w)
        win_qx = np.lib.stride_tricks.sliding_window_view(q_x, w)
        win_eff = np.lib.stride_tricks.sliding_window_view(det_eff, w)
        win_sz = np.lib.stride_tricks.sliding_window_view(sift_z, w)
        win_sx = np.lib.stride_tricks.sliding_window_view(sift_x, w)
        win_sift = np.lib.stride_tricks.sliding_window_view(sift_len, w)
        win_raw = np.lib.stride_tricks.sliding_window_view(raw_photons, w)
        win_gmean = np.lib.stride_tricks.sliding_window_view(gap_mean, w)
        win_gstd = np.lib.stride_tricks.sliding_window_view(gap_std, w)
        win_gcv = np.lib.stride_tricks.sliding_window_view(gap_cv, w)
        win_clust = np.lib.stride_tricks.sliding_window_view(clustering, w)

        end_idx = np.arange(w - 1, num_blocks)
        k_windows = len(win_q)

        # -------------------------------------------------------------
        # 1. Statistical Features (7)
        # -------------------------------------------------------------
        qber_mean = win_q.mean(axis=1)
        qber_std = win_q.std(axis=1, ddof=1)
        qber_variance = qber_std ** 2
        has_spread = qber_std > 1e-8
        with np.errstate(all="ignore"):
            qber_skewness = np.where(has_spread, stats.skew(win_q, axis=1, bias=False), 0.0)
            qber_kurtosis = np.where(has_spread, stats.kurtosis(win_q, axis=1, bias=False), 0.0)
        qber_iqr = np.percentile(win_q, 75, axis=1) - np.percentile(win_q, 25, axis=1)
        qber_cv = qber_std / (qber_mean + 1e-9)

        # -------------------------------------------------------------
        # 2. Burst & Instability Features (10)
        # -------------------------------------------------------------
        spike_density = np.mean(win_q > (qber_mean[:, None] + 2.0 * qber_std[:, None]), axis=1)

        # Vectorized burst duration: max consecutive blocks above mean + std
        above_thresh = (win_q > (qber_mean[:, None] + qber_std[:, None])).astype(int)
        burst_duration = np.zeros(k_windows, dtype=np.float64)
        for i in range(k_windows):
            row = above_thresh[i]
            cur_run, max_run = 0, 0
            for val in row:
                if val == 1:
                    cur_run += 1
                    if cur_run > max_run:
                        max_run = cur_run
                else:
                    cur_run = 0
            burst_duration[i] = max_run

        diffs_q = np.diff(win_q, axis=1)
        jump_energy = np.sum(diffs_q ** 2, axis=1)

        x_centered = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
        x_dot = np.dot(x_centered, x_centered) + 1e-12
        temporal_drift_slope = (win_q @ x_centered) / x_dot

        max_jump = np.max(np.abs(diffs_q), axis=1)
        jump_count = np.sum(np.abs(diffs_q) > (1.5 * qber_std[:, None]), axis=1).astype(np.float64)
        error_gap_mean = win_gmean.mean(axis=1)
        error_gap_std = win_gstd.mean(axis=1)
        error_gap_cv = win_gcv.mean(axis=1)
        burstiness_index = np.clip((qber_std - qber_mean) / (qber_std + qber_mean + 1e-9), -1.0, 1.0)

        # -------------------------------------------------------------
        # 3. Spectral & Temporal Features (12)
        # -------------------------------------------------------------
        centered_q = win_q - qber_mean[:, None]
        denom_ac = np.sum(centered_q ** 2, axis=1) + 1e-12
        autocorr_lag_1 = np.where(has_spread, np.sum(centered_q[:, :-1] * centered_q[:, 1:], axis=1) / denom_ac, 0.0)
        autocorr_lag_2 = np.where(has_spread, np.sum(centered_q[:, :-2] * centered_q[:, 2:], axis=1) / denom_ac, 0.0)
        autocorr_lag_3 = np.where(has_spread, np.sum(centered_q[:, :-3] * centered_q[:, 3:], axis=1) / denom_ac, 0.0)
        autocorr_lag_4 = np.where(has_spread, np.sum(centered_q[:, :-4] * centered_q[:, 4:], axis=1) / denom_ac, 0.0)
        autocorr_lag_5 = np.where(has_spread, np.sum(centered_q[:, :-5] * centered_q[:, 5:], axis=1) / denom_ac, 0.0)

        # Normalized DFT energy spectrum (rfft excluding DC bin 0)
        rfft_vals = np.fft.rfft(centered_q, axis=1)[:, 1:]
        mag_sq = np.abs(rfft_vals) ** 2
        tot_power = np.sum(mag_sq, axis=1, keepdims=True) + 1e-12
        p_spec = mag_sq / tot_power
        n_bins = p_spec.shape[1]
        b1 = max(1, n_bins // 3)
        b2 = max(b1 + 1, 2 * n_bins // 3)
        dft_low_energy = np.sum(p_spec[:, :b1], axis=1)
        dft_mid_energy = np.sum(p_spec[:, b1:b2], axis=1)
        dft_high_energy = np.sum(p_spec[:, b2:], axis=1)

        with np.errstate(all="ignore"):
            log_p = np.where(p_spec > 1e-12, np.log2(np.where(p_spec > 1e-12, p_spec, 1.0)), 0.0)
            spectral_entropy = -np.sum(p_spec * log_p, axis=1)
        k_indices = np.arange(1, n_bins + 1, dtype=np.float64)
        spectral_centroid = np.sum(p_spec * k_indices[None, :], axis=1) / (np.sum(p_spec, axis=1) + 1e-12)

        # Shannon entropy of binned QBER values in window
        edges = np.linspace(0.0, 0.35, 6)
        bin_idx = np.searchsorted(edges, win_q, side="right") - 1
        bin_idx[win_q == edges[-1]] = 4
        in_range = (win_q >= edges[0]) & (win_q <= edges[-1])
        counts = np.stack([np.sum((bin_idx == k) & in_range, axis=1) for k in range(5)], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            probs = counts / counts.sum(axis=1, keepdims=True)
            plogp = np.where(probs > 0, probs * np.log2(np.where(probs > 0, probs, 1.0)), 0.0)
        shannon_entropy = -plogp.sum(axis=1)

        e_t = win_q[:, -1]
        ema_divergence = e_t - ema[end_idx]

        # -------------------------------------------------------------
        # 4. Basis-Dependent Features (12)
        # -------------------------------------------------------------
        qber_rectilinear = win_qz.mean(axis=1)
        qber_diagonal = win_qx.mean(axis=1)
        basis_asymmetry = np.abs(qber_rectilinear - qber_diagonal)
        basis_ratio = qber_rectilinear / (qber_diagonal + 1e-6)

        basis_burst_energy_z = np.sum(np.diff(win_qz, axis=1) ** 2, axis=1)
        basis_burst_energy_x = np.sum(np.diff(win_qx, axis=1) ** 2, axis=1)

        var_qz = win_qz.var(axis=1, ddof=1)
        var_qx = win_qx.var(axis=1, ddof=1)
        basis_variance_ratio = var_qz / (var_qx + 1e-9)

        # Cross correlation between basis sequences
        centered_qz = win_qz - qber_rectilinear[:, None]
        centered_qx = win_qx - qber_diagonal[:, None]
        cov_zx = np.sum(centered_qz * centered_qx, axis=1)
        std_zx = np.sqrt(np.sum(centered_qz**2, axis=1) * np.sum(centered_qx**2, axis=1)) + 1e-12
        basis_cross_correlation = np.where((var_qz > 1e-8) & (var_qx > 1e-8), cov_zx / std_zx, 0.0)

        with np.errstate(all="ignore"):
            skew_z = np.where(var_qz > 1e-8, stats.skew(win_qz, axis=1, bias=False), 0.0)
            skew_x = np.where(var_qx > 1e-8, stats.skew(win_qx, axis=1, bias=False), 0.0)
            kurt_z = np.where(var_qz > 1e-8, stats.kurtosis(win_qz, axis=1, bias=False), 0.0)
            kurt_x = np.where(var_qx > 1e-8, stats.kurtosis(win_qx, axis=1, bias=False), 0.0)
        basis_skewness_diff = skew_z - skew_x
        basis_kurtosis_diff = kurt_z - kurt_x

        mean_sz = win_sz.mean(axis=1)
        mean_sx = win_sx.mean(axis=1)
        sift_ratio_z = mean_sz / (mean_sz + mean_sx + 1e-9)
        sift_ratio_x = 1.0 - sift_ratio_z

        # -------------------------------------------------------------
        # 5. Channel Interaction & Multi-Scale Features (22)
        # -------------------------------------------------------------
        key_generation_rate = win_sift.mean(axis=1) / (win_raw.mean(axis=1) + 1e-9)
        transmission_efficiency = win_eff.mean(axis=1)
        qber_loss_interaction = qber_mean / (transmission_efficiency + 1e-6)
        qber_loss_product = qber_mean * (1.0 - transmission_efficiency)
        loss_anomaly_index = transmission_efficiency - 0.50
        dark_count_ratio = 1e-5 / (qber_mean + 1e-6)

        # Multi-scale sub-windows: W=10, 25, 50
        multiscale_mean_w10 = win_q[:, -10:].mean(axis=1)
        multiscale_mean_w25 = win_q[:, -25:].mean(axis=1)
        multiscale_mean_w50 = qber_mean

        multiscale_std_w10 = win_q[:, -10:].std(axis=1, ddof=1)
        multiscale_std_w25 = win_q[:, -25:].std(axis=1, ddof=1)
        multiscale_std_w50 = qber_std

        multiscale_fano_w10 = (multiscale_std_w10 ** 2) / (multiscale_mean_w10 + 1e-9)
        multiscale_fano_w25 = (multiscale_std_w25 ** 2) / (multiscale_mean_w25 + 1e-9)
        multiscale_fano_w50 = (multiscale_std_w50 ** 2) / (multiscale_mean_w50 + 1e-9)

        x10 = np.arange(10, dtype=np.float64) - 4.5
        x25 = np.arange(25, dtype=np.float64) - 12.0
        multiscale_drift_w10 = (win_q[:, -10:] @ x10) / (np.dot(x10, x10) + 1e-12)
        multiscale_drift_w25 = (win_q[:, -25:] @ x25) / (np.dot(x25, x25) + 1e-12)
        multiscale_drift_w50 = temporal_drift_slope

        error_clustering_coefficient = win_clust.mean(axis=1)
        loss_gradient = (win_eff @ x_centered) / x_dot

        # Efficiency entropy
        eff_edges = np.linspace(0.0, 1.0, 6)
        eff_bin = np.searchsorted(eff_edges, win_eff, side="right") - 1
        eff_bin[win_eff == eff_edges[-1]] = 4
        eff_counts = np.stack([np.sum(eff_bin == k, axis=1) for k in range(5)], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff_p = eff_counts / eff_counts.sum(axis=1, keepdims=True)
            efficiency_entropy = -np.sum(np.where(eff_p > 0, eff_p * np.log2(np.where(eff_p > 0, eff_p, 1.0)), 0.0), axis=1)

        qber_stability_index = np.exp(-np.abs(qber_mean - self.base_qber) / (self.base_qber + 1e-6))

        # Assemble DataFrame
        t_index = end_idx + 1
        target_cls = int(seq.attack_class)
        is_attack = ((t_index >= seq.attack_start_block) & (target_cls > 0)).astype(int)
        cur_block_cls = np.where(t_index >= seq.attack_start_block, target_cls, 0)

        df = pd.DataFrame({
            "session_id": int(seq.session_id),
            "block_index": t_index,
            "channel_type": str(seq.channel_type),
            "session_attack_class": target_cls,
            "session_p_eve": float(seq.p_eve),
            "attack_start_block": int(seq.attack_start_block),
            "is_attack_active": is_attack,
            "block_class": cur_block_cls,
        })

        feature_arrays = [
            qber_mean, qber_variance, qber_std, qber_skewness, qber_kurtosis, qber_iqr, qber_cv,
            spike_density, burst_duration, jump_energy, temporal_drift_slope, max_jump, jump_count,
            error_gap_mean, error_gap_std, error_gap_cv, burstiness_index,
            autocorr_lag_1, autocorr_lag_2, autocorr_lag_3, autocorr_lag_4, autocorr_lag_5,
            dft_low_energy, dft_mid_energy, dft_high_energy, spectral_entropy, spectral_centroid,
            shannon_entropy, ema_divergence,
            qber_rectilinear, qber_diagonal, basis_asymmetry, basis_ratio,
            basis_burst_energy_z, basis_burst_energy_x, basis_variance_ratio,
            basis_cross_correlation, basis_skewness_diff, basis_kurtosis_diff,
            sift_ratio_z, sift_ratio_x,
            key_generation_rate, transmission_efficiency, qber_loss_interaction, qber_loss_product,
            loss_anomaly_index, dark_count_ratio,
            multiscale_mean_w10, multiscale_mean_w25, multiscale_mean_w50,
            multiscale_std_w10, multiscale_std_w25, multiscale_std_w50,
            multiscale_fano_w10, multiscale_fano_w25, multiscale_fano_w50,
            multiscale_drift_w10, multiscale_drift_w25, multiscale_drift_w50,
            error_clustering_coefficient, loss_gradient, efficiency_entropy, qber_stability_index
        ]

        assert len(feature_arrays) == 63, f"Expected 63 arrays, got {len(feature_arrays)}"
        for fname, farr in zip(self.feature_names, feature_arrays):
            df[fname] = np.asarray(farr, dtype=np.float64)

        return df
