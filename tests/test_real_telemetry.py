"""Real-telemetry loader: feature definitions must match the simulated schema."""

from pathlib import Path

import numpy as np
import pytest

from src.data.real_telemetry import ORDER_FREE_FEATURES, load_blocks, order_free_features
from src.features.time_series import CausalMultiAttackFeatureExtractor
from src.simulator.bb84_engine import BB84Simulator

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def test_order_free_features_match_the_extractor():
    seq = BB84Simulator(raw_photons_per_block=2048, random_state=1).simulate_session(
        1, 1, attack_mode="multiattack", blocks_per_session=60, attack_start_block=61)
    df = CausalMultiAttackFeatureExtractor(window_size=50).extract_from_session(seq)
    last = df.iloc[-1]
    ref = order_free_features(seq.block_qbers[-50:])
    for f in ORDER_FREE_FEATURES:
        assert np.isclose(ref[f], last[f], rtol=1e-9, atol=1e-12), f


def test_order_free_features_ignore_record_order():
    q = np.random.default_rng(0).normal(0.04, 0.003, 60)
    a = order_free_features(q)
    b = order_free_features(np.sort(q))
    assert all(np.isclose(a[k], b[k]) for k in ORDER_FREE_FEATURES)


@pytest.mark.skipif(not any(RAW.glob("*full_sweep.csv")), reason="Zenodo 21792844 CSVs not present")
def test_loader_reads_local_zenodo_files():
    blocks = load_blocks(RAW)
    assert len(blocks) > 0
    assert blocks["qber_mean"].between(0.0, 0.5).all()
    assert set(ORDER_FREE_FEATURES) <= set(blocks.columns)
