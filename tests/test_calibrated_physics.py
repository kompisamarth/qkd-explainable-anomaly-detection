"""Physics sanity checks for the calibrated WCP simulator. The pipeline fails if these break."""

import copy

import numpy as np
import pytest

from src.config import load_physics
from src.evaluation.finite_size import clopper_pearson_interval, wilson_interval
from src.simulator.calibrated import (
    CLASS_KEYS, CalibratedBB84Simulator, SessionPhysics, coherent_clicks, fock_clicks,
)

BASE = load_physics()["calibrated_bb84"]


def quiet_physics(**over):
    """A static channel: no drift, no loss steps, no afterpulsing or dead time."""
    p = copy.deepcopy(BASE)
    p["benign_dynamics"].update(misalignment_drift_std=0.0, loss_step_prob=0.0, efficiency_drift_std=0.0)
    p["detector"].update(enable_afterpulsing=False, enable_dead_time=False)
    p["source"]["pulses_per_block"] = 2_000_000
    p.update(over)
    return p


def fixed_session(attack="normal", intensity=0.0, e_d=0.0, p_dark=0.0, t_db=3.0, eta=0.6, **atk):
    return SessionPhysics(
        channel_type="fiber", distance_km=10.0, loss_db=t_db, detector_type="snspd",
        eta=np.full((2, 2), eta), p_dark=p_dark, afterpulse=0.0, dead_time_s=0.0, e_d0=e_d,
        intensities=np.array([0.5]), intensity_probs=np.array([1.0]),
        attack=attack, intensity=intensity, attack_params=dict(atk),
    )


def run(sp, blocks=20, seed=0, **phys_over):
    sim = CalibratedBB84Simulator(quiet_physics(**phys_over), random_state=seed)
    cls = CLASS_KEYS.index(sp.attack)
    return sim.simulate_session(1, cls, blocks_per_session=blocks, attack_start_block=1, physics=sp)


def pooled_qber(seq):
    return seq.block_errors.sum() / seq.block_sifted_lens.sum()


def test_no_eve_zero_noise_gives_exactly_zero_qber():
    seq = run(fixed_session())
    assert seq.block_sifted_lens.sum() > 10_000
    assert seq.block_errors.sum() == 0


def test_full_intercept_resend_gives_quarter_qber():
    seq = run(fixed_session("intercept_resend", intensity=1.0))
    n = seq.block_sifted_lens.sum()
    q = pooled_qber(seq)
    tol = 4 * np.sqrt(0.25 * 0.75 / n) + 0.002   # 4 sigma + double-click allowance
    assert abs(q - 0.25) < tol, f"QBER {q:.4f}, n={n}"


@pytest.mark.parametrize("e_d", [0.01, 0.02])
def test_misalignment_only_gives_qber_close_to_e_d(e_d):
    # Long, lossy link so double clicks are negligible; QBER should equal e_d
    seq = run(fixed_session(e_d=e_d, t_db=20.0))
    n = seq.block_sifted_lens.sum()
    q = pooled_qber(seq)
    assert abs(q - e_d) < 4 * np.sqrt(e_d * (1 - e_d) / n) + 5e-4, f"QBER {q:.5f} vs e_d {e_d}"


def test_dark_counts_alone_give_half_qber():
    d, err = coherent_clicks(0.0, 0.0, 1e-5, 1e-5)
    assert abs(err / d - 0.5) < 1e-6


def test_coherent_and_fock_agree_for_one_photon_at_low_efficiency():
    # A one-photon Fock state and a weak coherent pulse match to first order
    d_f, _ = fock_clicks(1, 0.01, 0.0, 0.0, 0.0)
    d_c, _ = coherent_clicks(0.01, 0.0, 0.0, 0.0)
    assert abs(d_f - d_c) < 1e-4


def test_pns_is_gain_matched_and_adds_no_error():
    honest = run(fixed_session(e_d=0.015, t_db=15.0, p_dark=1e-7), seed=1)
    pns = run(fixed_session("pns", intensity=1.0, e_d=0.015, t_db=15.0, p_dark=1e-7, gain_mismatch=0.0), seed=1)
    g_h, g_p = honest.block_transmission_effs.mean(), pns.block_transmission_effs.mean()
    assert abs(g_p / g_h - 1) < 0.02
    assert abs(pooled_qber(pns) - pooled_qber(honest)) < 0.002


def test_blinding_lowers_qber_below_misalignment():
    seq = run(fixed_session("detector_blinding", intensity=1.0, e_d=0.015, residual_qber=0.003, gain_mismatch=0.0))
    assert pooled_qber(seq) < 0.006


def test_finite_size_intervals_contain_estimate_and_shrink():
    lo1, hi1 = wilson_interval(10, 100)
    lo2, hi2 = wilson_interval(100, 1000)
    assert lo1 < 0.1 < hi1 and lo2 < 0.1 < hi2
    assert (hi2 - lo2) < (hi1 - lo1)
    lo, hi = clopper_pearson_interval(np.array([0, 5]), np.array([50, 50]))
    assert lo[0] == 0.0 and hi[0] > 0 and lo[1] < 0.1 < hi[1]


def test_every_class_draws_the_channel_from_one_distribution():
    sim = CalibratedBB84Simulator(BASE, random_state=0)
    draws = {}
    for cls, key in enumerate(CLASS_KEYS):
        rng = np.random.default_rng(123)
        sp = sim.draw_session_physics(rng, key)
        draws[key] = (sp.distance_km, sp.loss_db, sp.detector_type, sp.p_dark, sp.e_d0, sp.loss_step_block)
    assert len(set(draws.values())) == 1, "channel draws depend on the class"


def test_same_seed_reproduces_session():
    sim_a = CalibratedBB84Simulator(BASE, random_state=7)
    sim_b = CalibratedBB84Simulator(BASE, random_state=7)
    a = sim_a.simulate_session(1, 1, blocks_per_session=60, attack_start_block=30, session_seed=99)
    b = sim_b.simulate_session(1, 1, blocks_per_session=60, attack_start_block=30, session_seed=99)
    np.testing.assert_array_equal(a.block_qbers, b.block_qbers)
