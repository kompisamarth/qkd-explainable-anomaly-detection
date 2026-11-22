"""Loader for real QKD telemetry: Zenodo 21792844 (TCD / CONNECT testbed, CC-BY-4.0).

Horgan J., Quigley N., Hajar B., Briantcev D., Kaszubowska-Anandarajah A., Kilper D.,
"QKD Network Characterisation and Coexistence Datasets with Standardised Performance
Metrics and Experimental Methods", Zenodo (2026), doi:10.5281/zenodo.21792844.

The CSVs hold one row per QKD measurement record with at least `VOA_dB`, `QBER` and
`SecureKeyRate_bps` (some files add noise-injection power and fibre length). In the
coexistence files rows are sorted by QBER within each attenuation setting and carry no
timestamp, so only ORDER-INDEPENDENT statistics are meaningful. Each (file, noise, fibre,
VOA) group becomes one "window" with the statistical features below, mapped to the same
names as the simulated feature schema.
"""

import json
import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

ZENODO_RECORD = 21792844
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
CITATION = ("Horgan J. et al., QKD Network Characterisation and Coexistence Datasets with "
            "Standardised Performance Metrics and Experimental Methods, Zenodo (2026), "
            "doi:10.5281/zenodo.21792844 (CC-BY-4.0)")

# Features that do not depend on record order, so they can be computed from the real CSVs
ORDER_FREE_FEATURES = ["qber_mean", "qber_variance", "qber_std", "qber_skewness", "qber_kurtosis",
                       "qber_iqr", "qber_cv", "shannon_entropy"]


def download(dest: Path) -> List[Path]:
    """Fetch every file of the Zenodo record into `dest` (about 2.2 MB)."""
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ZENODO_API, timeout=30) as r:
        rec = json.load(r)
    out = []
    for f in rec["files"]:
        target = dest / f["key"]
        if not target.exists():
            urllib.request.urlretrieve(f["links"]["self"], target)
        out.append(target)
    return out


def order_free_features(q: np.ndarray) -> Dict[str, float]:
    """Same definitions as CausalMultiAttackFeatureExtractor for the order-free subset."""
    q = np.asarray(q, dtype=np.float64)
    mean = float(q.mean())
    std = float(q.std(ddof=1)) if len(q) > 1 else 0.0
    spread = std > 1e-8 and len(q) >= 4
    hist, _ = np.histogram(q, bins=5, range=(0.0, 0.35))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return {
        "qber_mean": mean,
        "qber_variance": std ** 2,
        "qber_std": std,
        "qber_skewness": float(stats.skew(q, bias=False)) if spread else 0.0,
        "qber_kurtosis": float(stats.kurtosis(q, bias=False)) if spread else 0.0,
        "qber_iqr": float(np.percentile(q, 75) - np.percentile(q, 25)),
        "qber_cv": std / (mean + 1e-9),
        "shannon_entropy": float(-(p * np.log2(p)).sum()),
    }


def load_blocks(raw_dir: Path, min_records: int = 10) -> pd.DataFrame:
    """One row per (file, noise, fibre, VOA) group with order-free features."""
    rows = []
    for csv in sorted(Path(raw_dir).rglob("*.csv")):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if not {"VOA_dB", "QBER"}.issubset(df.columns):
            continue
        keys = [c for c in ("Noise_dBm", "Sweep_km", "VOA_dB") if c in df.columns]
        for gk, g in df.groupby(keys, dropna=False):
            q = pd.to_numeric(g["QBER"], errors="coerce").dropna().to_numpy()
            q = q[(q >= 0) & (q < 0.5)]
            if len(q) < min_records:
                continue
            gk = gk if isinstance(gk, tuple) else (gk,)
            meta = dict(zip(keys, gk))
            skr = pd.to_numeric(g.get("SecureKeyRate_bps", pd.Series(dtype=float)), errors="coerce")
            rows.append({"source_file": csv.name, **meta, "records": int(len(q)),
                         "mean_skr_bps": float(skr.mean()) if len(skr) else np.nan,
                         **order_free_features(q)})
    return pd.DataFrame(rows)


def summary_qber_vs_attenuation(raw_dir: Path) -> pd.DataFrame:
    """Per-VOA mean QBER from the dataset's own *_stats.json summaries."""
    rows = []
    for js in sorted(Path(raw_dir).rglob("*stats.json")):
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            continue
        files = data.values() if isinstance(data, dict) and "voa_blocks" not in data else [data]
        for entry in files:
            if not isinstance(entry, dict) or "voa_blocks" not in entry:
                continue
            for voa, blk in entry["voa_blocks"].items():
                qs = blk.get("qber_stats", {})
                if "mean" in qs:
                    rows.append({"source": entry.get("file_name", js.name), "voa_db": float(voa),
                                 "mean_qber": float(qs["mean"]), "std_qber": float(qs.get("std_dev", np.nan)),
                                 "records": int(qs.get("count", 0))})
    return pd.DataFrame(rows).drop_duplicates()
