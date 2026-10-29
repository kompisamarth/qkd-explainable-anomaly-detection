"""Phase 2: Unit and regression tests for multi-attack physics, 63 features, and XGBoost."""

import time
import pytest
import numpy as np
import pandas as pd

from src.simulator.bb84_engine import BB84Simulator
from src.simulator.noise_models import FiberNoiseModel
from src.simulator.eve_attacks import EveAttacker
from src.features.time_series import (
    CausalMultiAttackFeatureExtractor,
    FEATURE_NAMES_63,
    FEATURE_GROUPS_63,
)


def test_fractional_ir_qber_bounds():
    """Verify Fractional IR empirical QBER matches theoretical formulation."""
    sim = BB84Simulator(raw_photons_per_block=200000, random_state=1)
    fiber = FiberNoiseModel(base_qber=0.03, random_state=101)
    p_eve = 0.20
    eve = EveAttacker(class_id=1, p_eve=p_eve, attack_type="intercept_resend", random_state=202)

    res = sim.simulate_block_vectorized(1, fiber, eve)
    theo_qber = EveAttacker.theoretical_qber(0.03, p_eve)
    assert abs(res.qber - theo_qber) < 0.004, f"Empirical {res.qber} deviated from theoretical {theo_qber}"


def test_pns_zero_qber_disturbance():
    """Verify PNS induces zero added bit error rate (Delta QBER ~ 0) while yield drops."""
    sim = BB84Simulator(raw_photons_per_block=50000, random_state=42)
    fiber = FiberNoiseModel(base_qber=0.03, random_state=42)
    eve_pns = EveAttacker(class_id=2, attack_type="pns", mu_wcp=0.30, random_state=42)

    res = sim.simulate_block_vectorized(1, fiber, eve_pns)
    # QBER should match baseline noise (0.03) within statistical margin
    assert abs(res.qber - 0.03) < 0.006, f"PNS QBER {res.qber} unexpectedly disturbed"
    # Detection count must drop significantly compared to full raw block
    assert res.detection_count < res.raw_photons * 0.35, f"PNS detection count {res.detection_count} did not show multi-photon attenuation"


def test_fake_state_click_statistics():
    """Verify Fake State blinding produces near-zero QBER."""
    sim = BB84Simulator(raw_photons_per_block=50000, random_state=42)
    fiber = FiberNoiseModel(base_qber=0.03, random_state=42)
    eve_fake = EveAttacker(class_id=3, attack_type="fake_state", random_state=42)

    res = sim.simulate_block_vectorized(1, fiber, eve_fake)
    assert res.qber < 0.012, f"Fake State QBER {res.qber} exceeded classical linear mode threshold"
    assert res.sifted_bits > 0, "Fake State should produce sifted basis-coincident clicks"


def test_time_shift_basis_asymmetry():
    """Verify Time-Shift attack induces severe basis-dependent asymmetry (|q_Z - q_X| >> 0)."""
    sim = BB84Simulator(raw_photons_per_block=50000, random_state=42)
    fiber = FiberNoiseModel(base_qber=0.03, random_state=42)
    eve_ts = EveAttacker(class_id=4, attack_type="time_shift", shift_delay_prob=0.75, random_state=42)

    res = sim.simulate_block_vectorized(1, fiber, eve_ts)
    asymm = abs(res.qber_z - res.qber_x)
    assert asymm > 0.05, f"Time-Shift asymmetry {asymm} was insufficient (expected > 0.05)"
    # Overall QBER must stay below 11%
    assert res.qber < 0.11, f"Time-Shift QBER {res.qber} unexpectedly breached 11% threshold"


def test_63_feature_matrix_dimensions():
    """Verify extractor extracts exactly 63 physics features across all 5 groups."""
    assert len(FEATURE_NAMES_63) == 63
    sim = BB84Simulator(raw_photons_per_block=1024, random_state=42)
    seq = sim.simulate_session(1, 1, attack_mode="multiattack", blocks_per_session=60, attack_start_block=30)
    extractor = CausalMultiAttackFeatureExtractor(window_size=50)
    df = extractor.extract_from_session(seq)

    assert df.shape[1] == 63 + 8  # 63 features + 8 metadata columns
    for f in FEATURE_NAMES_63:
        assert f in df.columns, f"Feature {f} missing from extracted DataFrame"
    assert not df[FEATURE_NAMES_63].isna().any().any(), "Features contain NaN values"


def test_multiattack_feature_causality():
    """Verify that perturbing future blocks does not alter past feature rows."""
    sim = BB84Simulator(random_state=42)
    seq = sim.simulate_session(1, 2, attack_mode="multiattack", blocks_per_session=100, attack_start_block=50)
    extractor = CausalMultiAttackFeatureExtractor(window_size=50)

    df_clean = extractor.extract_from_session(seq)

    # Clone sequence and corrupt block 80 onward
    from copy import deepcopy
    seq_corrupted = deepcopy(seq)
    seq_corrupted.block_qbers[80:] = 0.45
    seq_corrupted.block_qbers_z[80:] = 0.45

    df_corrupted = extractor.extract_from_session(seq_corrupted)

    # Feature rows ending before block 80 must be bit-for-bit identical
    before_mask = df_clean["block_index"] < 80
    np.testing.assert_array_equal(
        df_clean.loc[before_mask, FEATURE_NAMES_63].values,
        df_corrupted.loc[before_mask, FEATURE_NAMES_63].values,
    )


def test_nuisance_randomization_breaks_per_class_fingerprints():
    """Transmission efficiency must overlap across all classes (README P2.1).

    Before the fix, every class ran at a fixed channel transmittance of 1.0, so
    this feature was exactly 1.000 with zero variance for classes 0 and 1 and
    separated three of the five classes on its own.
    """
    sim = BB84Simulator(raw_photons_per_block=1024, random_state=5)
    ranges = {}
    for cls in range(5):
        effs = []
        for i in range(12):
            seq = sim.simulate_session(
                session_id=cls * 100 + i,
                target_attack_class=cls,
                blocks_per_session=60,
                attack_start_block=30,
                attack_mode="multiattack",
                randomize_nuisance=True,
            )
            effs.extend(seq.block_transmission_effs[30:].tolist())
        arr = np.asarray(effs)
        assert arr.std() > 0.02, f"class {cls} transmission efficiency is near-constant"
        ranges[cls] = (float(arr.min()), float(arr.max()))

    lo0, hi0 = ranges[0]
    for cls in range(1, 5):
        lo, hi = ranges[cls]
        assert lo < hi0 and lo0 < hi, f"class {cls} range {ranges[cls]} misses benign {ranges[0]}"
