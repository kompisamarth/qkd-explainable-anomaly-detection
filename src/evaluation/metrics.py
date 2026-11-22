"""Security-oriented metrics shared by the evaluation scripts.

FNR (missed attacks) is the primary security metric: a missed attack leaks key, a
false alarm only costs throughput.
"""

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def binary_rates(y_attack: Sequence[bool], alarm: Sequence[bool]) -> Dict[str, float]:
    y = np.asarray(y_attack, dtype=bool)
    a = np.asarray(alarm, dtype=bool)
    tp = int(np.sum(y & a)); fn = int(np.sum(y & ~a))
    fp = int(np.sum(~y & a)); tn = int(np.sum(~y & ~a))
    return {
        "fnr": fn / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def finite_size_threshold_alarm(df: pd.DataFrame, threshold: float) -> np.ndarray:
    """Alarm when the block's Wilson lower bound exceeds the threshold.

    A plain per-block QBER > 11% test fires on statistical fluctuations of small
    blocks; requiring the lower confidence bound to exceed 11% does not. Returns all
    False when the data carries no confidence interval (legacy datasets).
    """
    if "qber_ci_low" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df["qber_ci_low"].to_numpy(float) > threshold


def mean_ci(values: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    """Mean, sample std and a Student-t confidence interval over seeds."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    m = float(v.mean())
    s = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * s / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": m, "std": s, "ci95_low": m - half, "ci95_high": m + half, "n_seeds": n}


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> float:
    """ECE on the top-class confidence (Guo et al., ICML 2017)."""
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_curve(y_bin: np.ndarray, p_attack: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Binned P(attack) against the observed attack frequency."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p_attack, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "mean_predicted": float(p_attack[m].mean()),
                         "observed_frequency": float(y_bin[m].mean()), "count": int(m.sum())})
    return pd.DataFrame(rows)


def detection_latency(df: pd.DataFrame, alarm: np.ndarray, correct_class: np.ndarray = None) -> pd.DataFrame:
    """Per attacked session: blocks from onset to the first (correct) alarm, and the
    pre-onset false-alarm rate. `correct_class` (optional) requires the alarm to name
    the true class; otherwise any non-normal alarm counts.
    """
    d = df[["session_id", "block_index", "attack_start_block", "session_attack_class", "session_p_eve"]].copy()
    d["alarm"] = np.asarray(alarm, dtype=bool)
    if correct_class is not None:
        d["alarm"] &= np.asarray(correct_class, dtype=bool)
    rows = []
    for sid, g in d.groupby("session_id"):
        onset = int(g["attack_start_block"].iloc[0])
        cls = int(g["session_attack_class"].iloc[0])
        pre = g[g["block_index"] < onset]
        post = g[(g["block_index"] >= onset) & g["alarm"]]
        n_post = int((g["block_index"] >= onset).sum())
        rows.append({
            "session_id": sid, "class_id": cls, "intensity": float(g["session_p_eve"].iloc[0]),
            "pre_attack_false_alarm_rate": float(pre["alarm"].mean()) if len(pre) else np.nan,
            "detected": bool(len(post)) if cls > 0 else np.nan,
            "latency_blocks": float(post["block_index"].min() - onset) if (cls > 0 and len(post)) else
            (float(n_post) if cls > 0 else np.nan),
        })
    return pd.DataFrame(rows)
