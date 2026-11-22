"""Dataset generation: reproducibility, session-level splits, metadata and leakage audit."""

import json

import numpy as np
import pandas as pd

from src.data.dataset import export, generate_sessions, leakage_audit, model_features, session_split
from src.simulator.calibrated import CLASS_NAMES


def small(seed=3, **kw):
    return generate_sessions(2, seed, classes=[0, 1, 2], blocks=70, onset=36, window=30, ema=5, **kw)


def test_same_seed_is_identical_for_any_number_of_workers():
    a, _ = small(n_jobs=1)
    b, _ = small(n_jobs=2)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_data():
    a, _ = small(seed=3, n_jobs=1)
    b, _ = small(seed=4, n_jobs=1)
    assert not np.allclose(a["qber_mean"].values, b["qber_mean"].values)


def test_session_split_never_shares_sessions_and_keeps_every_class():
    df, _ = generate_sessions(5, 1, classes=[0, 1], blocks=60, onset=31, window=30, ema=5, n_jobs=1)
    tr, te = session_split(df, 0.4, seed=0)
    assert not set(tr["session_id"]) & set(te["session_id"])
    assert set(te["session_attack_class"]) == {0, 1}
    assert len(tr) + len(te) == len(df)


def test_heldout_uses_unseen_distances_and_intensities():
    _, meta_train = small(n_jobs=1)
    _, meta_held = small(heldout=True, n_jobs=1)
    from src.config import load_physics
    p = load_physics()["calibrated_bb84"]
    lo, hi = p["channel"]["fiber"]["distance_km"]
    hlo, hhi = p["heldout"]["distance_km"]
    assert meta_train["distance_km"].between(lo, hi).all()
    assert meta_held["distance_km"].between(hlo, hhi).all()
    ir = meta_held[meta_held["attack"] == "intercept_resend"]["intensity"]
    assert ir.between(*p["attacks"]["intercept_resend"]["heldout_intensity"]).all()


def test_export_writes_csv_parquet_and_sidecar(tmp_path):
    df, _ = small(n_jobs=1)
    paths = export(df, tmp_path / "toy", {"seed": 3})
    meta = json.loads((tmp_path / "toy.meta.json").read_text())
    for key in ("seed", "config_hash", "git_commit", "generated_utc", "block_class_counts", "session_class_counts"):
        assert key in meta
    assert pd.read_parquet(paths["parquet"]).shape == df.shape
    assert meta["rows"] == len(df)


def test_leakage_audit_reports_every_class_and_flags_a_planted_leak():
    df, _ = small(n_jobs=1)
    feats = model_features(df)
    clean = leakage_audit(df, feats, len(CLASS_NAMES))
    assert {"0", "1", "2"} <= set(clean)
    leaky = df.copy()
    leaky["planted"] = (leaky["block_class"] == 2).astype(float)
    out = leakage_audit(leaky, feats + ["planted"], len(CLASS_NAMES))
    assert out["2"]["flagged"] and out["2"]["feature"] == "planted"
