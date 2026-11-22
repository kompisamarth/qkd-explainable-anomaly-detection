"""Dataset generation, export and leakage auditing for the calibrated multi-attack task.

Design rules (audit L-1, L-2, L-7):
  * Every session gets its own seed from one SeedSequence, so a dataset is identical
    whether it is generated on 1 or 24 workers.
  * Channel, detector and noise values come from the same distribution for every class
    (see CalibratedBB84Simulator.draw_session_physics).
  * Splits are by session id; no session contributes rows to two splits.
  * Each exported file gets a sidecar `<name>.meta.json` with seed, config hash, git
    commit, date, class counts and the parameter ranges that produced it.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from src.config import config_hash, git_commit, load_physics
from src.features.time_series import FEATURE_NAMES_63, FEATURE_NAMES_DECOY, CausalMultiAttackFeatureExtractor
from src.simulator.calibrated import CLASS_NAMES, CalibratedBB84Simulator

LEAKAGE_AUC_ALERT = 0.95


def _one_session(physics, session_id, cls, seed, channel, blocks, onset, heldout, window, ema):
    sim = CalibratedBB84Simulator(physics, random_state=seed)
    seq = sim.simulate_session(session_id, cls, channel_type=channel, blocks_per_session=blocks,
                               attack_start_block=onset, heldout=heldout, session_seed=seed)
    df = CausalMultiAttackFeatureExtractor(window_size=window, ema_span=ema).extract_from_session(seq)
    meta = {"session_id": session_id, "class_id": cls, "session_seed": seed, "heldout": heldout,
            **seq.physics.summary()}
    return df, meta


def generate_sessions(
    sessions_per_class: int,
    seed: int,
    classes: Sequence[int],
    channel: str = "fiber",
    blocks: int = 150,
    onset: int = 76,
    heldout: bool = False,
    window: int = 50,
    ema: int = 10,
    first_session_id: int = 1,
    physics: Optional[Dict] = None,
    n_jobs: int = -1,
):
    """Returns (features DataFrame, per-session metadata DataFrame)."""
    physics = physics or load_physics()["calibrated_bb84"]
    jobs = [(c, i) for c in classes for i in range(sessions_per_class)]
    seeds = np.random.SeedSequence(seed).generate_state(len(jobs), dtype=np.uint32)
    res = Parallel(n_jobs=n_jobs, batch_size=8)(
        delayed(_one_session)(physics, first_session_id + j, c, int(seeds[j]), channel, blocks, onset,
                              heldout, window, ema)
        for j, (c, _) in enumerate(jobs))
    feats = pd.concat([r[0] for r in res], ignore_index=True)
    meta = pd.DataFrame([r[1] for r in res])
    return feats, meta


def model_features(df: pd.DataFrame) -> List[str]:
    """63 base features, plus the decoy group when the data carries it."""
    return FEATURE_NAMES_63 + [f for f in FEATURE_NAMES_DECOY if f in df.columns]


def leakage_audit(df: pd.DataFrame, features: Sequence[str], n_classes: int,
                  max_rows: int = 60000, seed: int = 0) -> Dict:
    """Best single-feature one-vs-rest AUC for every class (including Normal).

    A feature that alone separates a class with AUC > 0.95 is reported as possible
    leakage: physics rarely gives one clean fingerprint, simulators often do.
    """
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    y = df["block_class"].to_numpy(int)
    out = {}
    for cls in range(n_classes):
        yb = (y == cls).astype(int)
        if yb.sum() in (0, len(yb)):
            continue
        best, best_f = 0.0, ""
        for f in features:
            v = df[f].to_numpy(float)
            if not np.isfinite(v).all() or np.ptp(v) == 0:
                continue
            a = roc_auc_score(yb, v)
            a = max(a, 1 - a)
            if a > best:
                best, best_f = a, f
        out[str(cls)] = {"class_name": CLASS_NAMES[cls], "best_auc": float(best), "feature": best_f,
                         "flagged": bool(best > LEAKAGE_AUC_ALERT)}
    out["max_auc"] = float(max(v["best_auc"] for k, v in out.items() if k != "max_auc"))
    out["any_flagged"] = bool(any(v["flagged"] for k, v in out.items() if isinstance(v, dict)))
    return out


def export(df: pd.DataFrame, path_stem: Path, meta: Dict, parquet: bool = True) -> Dict[str, str]:
    """Write <stem>.csv, <stem>.parquet and <stem>.meta.json."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {"csv": str(path_stem.with_suffix(".csv"))}
    df.to_csv(paths["csv"], index=False)
    if parquet:
        paths["parquet"] = str(path_stem.with_suffix(".parquet"))
        df.to_parquet(paths["parquet"], index=False)
    full = {
        "file": path_stem.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config_hash": config_hash(),
        "rows": int(len(df)),
        "sessions": int(df["session_id"].nunique()),
        "block_class_counts": {str(k): int(v) for k, v in df["block_class"].value_counts().sort_index().items()},
        "session_class_counts": {str(k): int(v) for k, v in
                                 df.groupby("session_id")["session_attack_class"].first().value_counts().sort_index().items()},
        **meta,
    }
    with open(path_stem.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, default=str)
    return paths


def session_split(df: pd.DataFrame, test_frac: float, seed: int):
    """Stratified-by-class split of SESSION ids (never rows)."""
    rng = np.random.default_rng(seed)
    sess = df.groupby("session_id")["session_attack_class"].first()
    test_ids = []
    for cls, ids in sess.groupby(sess):
        ids = ids.index.to_numpy().copy()
        rng.shuffle(ids)
        test_ids.extend(ids[: int(round(len(ids) * test_frac))])
    is_test = df["session_id"].isin(set(test_ids))
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
