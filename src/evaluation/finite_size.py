"""Finite-size confidence intervals for a block's QBER.

A block with n sifted bits and k errors only estimates the true error rate. At the
~250-bit blocks of the original simulator the 95% interval is roughly +/- 2.5
percentage points, so comparing a single block against the asymptotic 11% line
over-trusts it. Both intervals below are vectorised over arrays of (k, n).
"""

import numpy as np
from scipy import stats


def wilson_interval(k, n, confidence: float = 0.95):
    """Wilson score interval (E. B. Wilson, JASA 22, 209, 1927). Returns (low, high)."""
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, 0.0)
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    low = np.where(n > 0, np.clip(centre - half, 0.0, 1.0), 0.0)
    high = np.where(n > 0, np.clip(centre + half, 0.0, 1.0), 1.0)
    return low, high


def clopper_pearson_interval(k, n, confidence: float = 0.95):
    """Exact binomial interval (Clopper & Pearson, Biometrika 26, 404, 1934). Returns (low, high)."""
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    a = 1.0 - confidence
    with np.errstate(invalid="ignore"):
        low = np.where(k > 0, stats.beta.ppf(a / 2, k, n - k + 1), 0.0)
        high = np.where(k < n, stats.beta.ppf(1 - a / 2, k + 1, n - k), 1.0)
    low = np.where(n > 0, low, 0.0)
    high = np.where(n > 0, high, 1.0)
    return low, high
