"""Regression tests for the BB84 simulator, causal features and baselines."""

import numpy as np
import pytest

from src.simulator.bb84_engine import BB84Simulator
from src.simulator.eve_attacks import EveAttacker
from src.simulator.noise_models import FiberNoiseModel, FreeSpaceNoiseModel
from src.features.time_series import CausalTimeSeriesExtractor, FEATURE_NAMES
from src.models.hoeffding_baseline import HoeffdingDetector


def test_noiseless_no_eve_gives_zero_qber():
    sim = BB84Simulator(raw_photons_per_block=20000, random_state=0)
    res = sim.simulate_block_vectorized(1, FiberNoiseModel(0.0, random_state=0), EveAttacker(0, 0.0, random_state=0))
    assert res.qber == 0.0
    assert 0.48 < res.sifted_bits / res.raw_photons < 0.52


@pytest.mark.parametrize("p_eve", [0.2, 1.0])
def test_intercept_resend_matches_theory(p_eve):
    sim = BB84Simulator(raw_photons_per_block=200000, random_state=1)
    fiber = FiberNoiseModel(0.03, random_state=101)
    res = sim.simulate_block_vectorized(1, fiber, EveAttacker(2, p_eve, random_state=202))
    assert abs(res.qber - EveAttacker.theoretical_qber(0.03, p_eve)) < 0.004


def test_same_seed_reproduces_and_new_seed_differs():
    a = BB84Simulator(random_state=7).simulate_session(1, 2, blocks_per_session=30, attack_start_block=10)
    b = BB84Simulator(random_state=7).simulate_session(1, 2, blocks_per_session=30, attack_start_block=10)
    c = BB84Simulator(random_state=8).simulate_session(1, 2, blocks_per_session=30, attack_start_block=10)
    np.testing.assert_array_equal(a.block_qbers, b.block_qbers)
    assert not np.array_equal(a.block_qbers, c.block_qbers)


def test_session_regime_shift_labels():
    seq = BB84Simulator(random_state=3).simulate_session(1, 1, blocks_per_session=40, attack_start_block=21)
    assert np.all(seq.block_classes[:20] == 0) and np.all(seq.block_classes[20:] == 1)
    assert seq.block_qbers[:20].mean() < 0.06 < 0.2 < seq.block_qbers[20:].mean()


def test_free_space_drift_is_observable_and_correlated():
    fso = FreeSpaceNoiseModel(base_qber=0.03, scintillation_std=0.30, ar1_phi=0.75, random_state=0)
    q = np.array([fso.get_block_qber(b) for b in range(5000)])
    assert abs(q.mean() - 0.03) < 0.002
    assert q.std() > np.sqrt(0.03 * 0.97 / 5120)  # exceeds windowed binomial noise
    assert np.corrcoef(q[:-1], q[1:])[0, 1] > 0.6


@pytest.mark.parametrize("attack_class,channel", [(0, "fiber"), (1, "fiber"), (2, "free_space")])
def test_vectorized_features_match_reference(attack_class, channel):
    seq = BB84Simulator(random_state=11).simulate_session(5, attack_class, channel_type=channel)
    ex = CausalTimeSeriesExtractor(window_size=10, ema_span=5)
    kwargs = dict(
        session_id=5, qbers=seq.block_qbers, sifted_lens=seq.block_sifted_lens,
        block_classes=seq.block_classes, attack_class=attack_class,
        channel_type=channel, p_eve=seq.p_eve,
    )
    fast = ex.extract_from_session(**kwargs)
    ref = ex.extract_from_session_reference(**kwargs)
    assert list(fast.columns) == list(ref.columns)
    np.testing.assert_allclose(fast[FEATURE_NAMES].values, ref[FEATURE_NAMES].values, rtol=0, atol=1e-12)
    assert (fast["is_attack_active"].values == ref["is_attack_active"].values).all()


def test_features_are_causal():
    """Changing a future block must not change any feature row that ends before it."""
    seq = BB84Simulator(random_state=4).simulate_session(1, 2, blocks_per_session=60, attack_start_block=30)
    ex = CausalTimeSeriesExtractor()
    q2 = seq.block_qbers.copy()
    q2[45:] = 0.5
    base = dict(session_id=1, sifted_lens=seq.block_sifted_lens, block_classes=seq.block_classes,
                attack_class=2, channel_type="fiber", p_eve=seq.p_eve, attack_start_block=30)
    a = ex.extract_from_session(qbers=seq.block_qbers, **base)
    b = ex.extract_from_session(qbers=q2, **base)
    before = a["block_index"] <= 45
    np.testing.assert_array_equal(a.loc[before, FEATURE_NAMES].values, b.loc[before, FEATURE_NAMES].values)


def test_hoeffding_false_positive_rate_bounded():
    det = HoeffdingDetector(base_qber=0.03, alpha=0.01, mode="single_block")
    rng = np.random.default_rng(0)
    n = 512
    qbers = rng.binomial(n, 0.03, size=20000) / n
    import pandas as pd
    X = pd.DataFrame({"qber_current": qbers, "sifted_bits": np.full(qbers.shape, n)})
    assert det.predict(X).mean() <= 0.01


def test_qiskit_measure_resend_matches_theory():
    pytest.importorskip("qiskit")
    sim = BB84Simulator(random_state=5)
    assert sim.verify_with_qiskit(num_photons=600, p_eve=0.0)["empirical_qber"] == 0.0
    res = sim.verify_with_qiskit(num_photons=3000, p_eve=1.0)
    assert abs(res["empirical_qber"] - 0.25) < 0.035
