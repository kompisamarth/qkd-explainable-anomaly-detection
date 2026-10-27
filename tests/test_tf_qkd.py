"""Physics checks for the SNS Twin-Field QKD model."""

import numpy as np

from src.simulator.tf_qkd import (
    ChannelParams, SNSIntensities, SNSTFQKD, SNSTimeSeriesSimulator,
    decoy_bb84_key_rate, plob_bound,
)


def test_ideal_phase_matched_interference_has_no_error():
    model = SNSTFQKD(ChannelParams(p_dark=0.0, phase_noise_rad=0.0, phase_slices=10**6))
    assert model.statistics(100, SNSIntensities())["E_X"] < 1e-9


def test_key_rate_scales_with_sqrt_eta():
    model = SNSTFQKD()
    _, a = model.optimize(200)
    _, b = model.optimize(300)
    # +100 km = +20 dB channel loss; sqrt scaling gives a factor of 10 (BB84 would give 100)
    assert 8 < a["key_rate"] / b["key_rate"] < 13


def test_beats_plob_and_bb84_at_long_distance_but_not_short():
    ch = ChannelParams()
    model = SNSTFQKD(ch)
    _, far = model.optimize(350)
    assert far["key_rate"] > plob_bound(350, ch)
    assert far["key_rate"] > decoy_bb84_key_rate(350, ch)[0]
    _, near = model.optimize(50)
    assert near["key_rate"] < plob_bound(50, ch)
    assert near["key_rate"] < decoy_bb84_key_rate(50, ch)[0]


def test_phase_tampering_is_invisible_in_z_but_visible_in_x():
    model = SNSTFQKD()
    x = SNSIntensities(eps=0.05, mu_z=0.45, mu_1=0.08, mu_2=0.3)
    clean = model.statistics(200, x)
    tampered = model.statistics(200, x, phase_tamper_frac=0.1)
    assert abs(clean["E_Z"] - tampered["E_Z"]) < 1e-12
    assert tampered["E_X"] > clean["E_X"] + 0.03
    assert tampered["key_rate"] < clean["key_rate"]


def test_time_series_is_reproducible():
    model = SNSTFQKD()
    x = SNSIntensities(eps=0.05, mu_z=0.45, mu_1=0.08, mu_2=0.3)
    a = SNSTimeSeriesSimulator(model, 200, x, 2e7, 5e6, random_state=3).simulate_session(True, 0.1, 40, 21)
    b = SNSTimeSeriesSimulator(model, 200, x, "2.0e7", "5.0e6", random_state=3).simulate_session(True, 0.1, 40, 21)
    np.testing.assert_array_equal(a.phase_err_x, b.phase_err_x)
    assert (a.block_classes[:20] == 0).all() and (a.block_classes[20:] == 2).all()
    assert a.phase_err_x[20:].mean() > a.phase_err_x[:20].mean()
