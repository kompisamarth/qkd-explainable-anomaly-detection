"""Physics checks for the MDI-QKD model and the detector-attack comparison."""

import numpy as np

from src.simulator.mdi_qkd import (
    MDIChannelParams,
    MDIQKD,
    bb84_blinding_attack,
    detector_attack_comparison,
    mdi_charlie_attack,
)
from src.simulator.tf_qkd import ChannelParams, SNSTFQKD, plob_bound


def test_ideal_relay_has_zero_z_basis_error():
    """A |psi-> announcement can only come from opposite Z bits, so E_Z is exactly 0."""
    m = MDIQKD(MDIChannelParams(p_dark=0.0, e_misalign=0.0, eta_det=1.0))
    st = m.key_rate(0.001, 0.3, 0.3)
    assert st["E_Z"] == 0.0
    assert st["e11_X"] == 0.0


def test_single_photon_phase_error_matches_misalignment_theory():
    """e11_X should equal 2 e_d (1 - e_d) once dark counts are negligible."""
    e_d = 0.015
    m = MDIQKD(MDIChannelParams(p_dark=0.0, e_misalign=e_d))
    _, e11 = m.single_photon_yield(0.5, 0.5)
    assert abs(e11 - 2 * e_d * (1 - e_d)) < 1e-12


def test_key_rate_scales_linearly_with_total_transmittance():
    """MDI-QKD rate follows eta_a * eta_b, so +100 km (20 dB) costs ~100x.

    This is the contrast with TF-QKD, which only loses ~10x over the same span.
    """
    m = MDIQKD()
    a = m.optimize(100)[1]["key_rate"]
    b = m.optimize(200)[1]["key_rate"]
    assert 50 < a / b < 200


def test_mdi_does_not_beat_plob_bound():
    """MDI-QKD buys security, not distance; it stays under the repeaterless bound."""
    ch = MDIChannelParams()
    m = MDIQKD(ch)
    for d in (100, 200, 300):
        rate = m.optimize(d)[1]["key_rate"]
        assert rate < plob_bound(d, ChannelParams(alpha_db_per_km=ch.alpha_db_per_km))


def test_tf_qkd_beats_mdi_at_long_distance():
    """Sanity cross-check between the two protocols already in the repo."""
    mdi = MDIQKD().optimize(300)[1]["key_rate"]
    sns = SNSTFQKD().optimize(300)[1]["key_rate"]
    assert sns > mdi


def test_blinding_gives_eve_the_bb84_key_with_no_alarm():
    """The attack this project exists because of: full key, no QBER rise."""
    r = bb84_blinding_attack(1.0, blinding_residual_qber=0.005)
    assert r["eve_information"] == 1.0
    assert r["observed_qber"] < 0.011          # far under the 11% abort threshold
    assert r["check_basis_error"] < 0.011      # Eve forges the check basis too
    assert r["protocol_aborts"] is False


def test_same_attack_on_mdi_is_loud_and_aborts():
    """Eve owning Charlie cannot hide: the X-basis phase error she cannot forge."""
    r = mdi_charlie_attack(1.0, e_z_honest=0.005, e_x_honest=0.030)
    assert r["observed_qber"] < 0.011          # Z basis still looks clean...
    assert r["check_basis_error"] > 0.40       # ...but the phase error is random
    assert r["protocol_aborts"] is True
    assert r["secure_key_rate_factor"] == 0.0  # no extractable key


def test_mdi_abort_threshold_is_crossed_at_modest_control():
    """Even partial control of Charlie trips the monitor."""
    rows = detector_attack_comparison(p_grid=np.linspace(0, 1, 21))
    first_abort = next(r["p_control"] for r in rows if r["mdi_aborts"])
    assert 0.0 < first_abort < 0.35
    # BB84 never aborts at any control fraction
    assert not any(r["bb84_aborts"] for r in rows)
