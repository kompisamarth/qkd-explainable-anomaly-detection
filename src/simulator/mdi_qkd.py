"""Measurement-Device-Independent QKD (MDI-QKD) simulator.

Why this protocol
-----------------
Of the attacks this project models, four live on the *detector* side: fake-state
detector blinding, detector-efficiency mismatch, time-shift and backflash. Those
are the hardest class to defend statistically, because blinding has been shown
experimentally to hand Eve the entire key *without raising the QBER* -- the one
alarm a conventional monitor listens for never rings.

MDI-QKD (Lo, Curty & Qi, PRL 108, 130503, 2012) removes that whole class by
construction. Alice and Bob both become transmitters; the Bell-state measurement
is performed by an untrusted relay, Charlie. Eve is *allowed* to own Charlie
completely. She still cannot learn the key, because:

  * the key bits come from Alice's and Bob's own state preparation, which Eve
    never touches, and
  * the security monitor -- the X-basis phase error -- is computed from states
    Alice and Bob prepared themselves and compared against Charlie's public
    announcement. Eve cannot forge it, because she does not know what they sent.

So in BB84 the adversary who owns the detectors owns the monitor too. In MDI-QKD
the monitor sits at the two honest parties. That difference is the entire point,
and `detector_attack_comparison()` below makes it quantitative.

What MDI-QKD does NOT buy you
-----------------------------
Its key rate scales linearly with the total transmittance eta_a * eta_b, so it
does NOT beat the PLOB bound and does NOT reach further than BB84 -- that is
TF-QKD's trick (see tf_qkd.py). MDI-QKD trades rate for security, not distance.
It also does nothing about source-side light injection (Trojan-horse, laser
seeding, induced photorefraction, laser damage), which remains open under every
protocol because every security proof assumes a sealed laboratory.

Model
-----
Charlie's station is a 50:50 beam splitter followed by a polarising beam
splitter in each output arm and four detectors (1H, 1V, 2H, 2V). Gains and error
rates for weak coherent pulses are computed from first principles: the input
amplitudes are combined on the beam splitter, the four detector mean photon
numbers are formed, click probabilities are 1 - (1 - p_d) exp(-n), and the
two-fold coincidence patterns that signal |psi-> are summed and averaged over
the random relative phase of the two independent lasers. This mirrors the
approach already used in tf_qkd.py.

Limitations: asymptotic (no finite-key correction), infinite-decoy limit for the
single-photon yield, and Y11/e11 are leading-order expressions. Documented in
the README limitations section.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


def binary_entropy(x: float) -> float:
    x = float(np.clip(x, 1e-15, 1 - 1e-15))
    return float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))


@dataclass(frozen=True)
class MDIChannelParams:
    """Fiber + relay detector parameters (typical MDI-QKD literature values)."""
    alpha_db_per_km: float = 0.2    # Standard SMF-28 attenuation
    eta_det: float = 0.6            # Charlie's detector efficiency
    p_dark: float = 1e-7            # Dark-count probability per detector per gate
    e_misalign: float = 0.015       # Polarization misalignment at Charlie
    f_ec: float = 1.16              # Error-correction inefficiency

    def arm_transmittance(self, distance_km: float) -> float:
        """Alice->Charlie (or Bob->Charlie) transmittance; Charlie sits midway."""
        return self.eta_det * 10 ** (-self.alpha_db_per_km * (distance_km / 2.0) / 10.0)

    def total_transmittance(self, distance_km: float) -> float:
        """End-to-end Alice<->Bob transmittance, for the PLOB comparison."""
        return 10 ** (-self.alpha_db_per_km * distance_km / 10.0)


class MDIQKD:
    """First-principles MDI-QKD model with a polarization-encoding BSM relay."""

    _PHASE_GRID = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)

    def __init__(self, channel: Optional[MDIChannelParams] = None):
        self.ch = channel or MDIChannelParams()

    # ---------- Charlie's Bell-state measurement ----------
    def _psi_minus_prob(self, n1h, n1v, n2h, n2v):
        """P(|psi-> announced) = P(1H and 2V, others silent) + P(1V and 2H, others silent)."""
        pd = self.ch.p_dark

        def click(n):
            return 1.0 - (1.0 - pd) * np.exp(-n)

        def silent(n):
            return (1.0 - pd) * np.exp(-n)

        pat_a = click(n1h) * click(n2v) * silent(n1v) * silent(n2h)
        pat_b = click(n1v) * click(n2h) * silent(n1h) * silent(n2v)
        return pat_a + pat_b

    def _mode_means(self, a_h, a_v, b_h, b_v):
        """Detector mean photon numbers after the 50:50 beam splitter."""
        n1h = np.abs(a_h + b_h) ** 2 / 2.0
        n2h = np.abs(a_h - b_h) ** 2 / 2.0
        n1v = np.abs(a_v + b_v) ** 2 / 2.0
        n2v = np.abs(a_v - b_v) ** 2 / 2.0
        return n1h, n1v, n2h, n2v

    def _prepare(self, amp, bit, basis, misalign_angle=0.0):
        """(H, V) amplitudes for a BB84 state of total amplitude `amp`."""
        if basis == 0:  # Z: H = bit 0, V = bit 1
            h, v = (amp, 0.0) if bit == 0 else (0.0, amp)
        else:           # X: (H +/- V)/sqrt(2)
            s = 1.0 if bit == 0 else -1.0
            h, v = amp / np.sqrt(2.0), s * amp / np.sqrt(2.0)
        if misalign_angle != 0.0:
            c, s_ = np.cos(misalign_angle), np.sin(misalign_angle)
            h, v = c * h - s_ * v, s_ * h + c * v
        return complex(h), complex(v)

    def wcp_stats(self, mu_a: float, mu_b: float, eta_a: float, eta_b: float, basis: int
                  ) -> Tuple[float, float]:
        """(gain, error rate) for weak coherent pulses in the given basis.

        In the Z basis a genuine |psi-> event requires Alice and Bob to have sent
        OPPOSITE bits, so Bob flips his bit; same-bit coincidences are errors.
        In the X basis |psi-> signals opposite bits as well.
        """
        a = np.sqrt(mu_a * eta_a)
        b = np.sqrt(mu_b * eta_b)
        theta = np.arcsin(np.sqrt(self.ch.e_misalign))

        gain_sum, err_sum = 0.0, 0.0
        for bit_a in (0, 1):
            for bit_b in (0, 1):
                a_h, a_v = self._prepare(a, bit_a, basis)
                b_h0, b_v0 = self._prepare(b, bit_b, basis, misalign_angle=theta)
                # independent lasers -> average over the random relative phase
                ph = np.exp(1j * self._PHASE_GRID)
                n1h, n1v, n2h, n2v = self._mode_means(a_h, a_v, b_h0 * ph, b_v0 * ph)
                p = float(np.mean(self._psi_minus_prob(n1h, n1v, n2h, n2v)))
                gain_sum += 0.25 * p
                if bit_a == bit_b:          # |psi-> should never fire on equal bits
                    err_sum += 0.25 * p
        gain = gain_sum
        return gain, (err_sum / gain if gain > 0 else 0.0)

    def single_photon_yield(self, eta_a: float, eta_b: float) -> Tuple[float, float]:
        """Leading-order (Y11, e11_X) for the single-photon pair contribution.

        Two indistinguishable single photons meeting on the beam splitter project
        onto |psi-> with probability 1/2 when their polarizations are orthogonal.
        Dark counts add an accidental floor.
        """
        pd = self.ch.p_dark
        y11 = 0.5 * eta_a * eta_b * (1.0 - pd) ** 2 + 4.0 * pd ** 2
        # X-basis error for single photons: misalignment on both arms, plus the
        # 1/2 random outcome on dark-count-driven accidentals.
        e_d = self.ch.e_misalign
        e_signal = 2.0 * e_d * (1.0 - e_d)
        acc = 4.0 * pd ** 2 / max(y11, 1e-300)
        e11 = e_signal * (1.0 - acc) + 0.5 * acc
        return float(y11), float(np.clip(e11, 0.0, 0.5))

    def key_rate(self, distance_km: float, mu_a: float, mu_b: float) -> Dict[str, float]:
        """Asymptotic MDI-QKD key rate (infinite decoys).

        R = Q11 [1 - H(e11_X)] - f Q_Z H(E_Z)
        """
        eta_a = eta_b = self.ch.arm_transmittance(distance_km)
        q_z, e_z = self.wcp_stats(mu_a, mu_b, eta_a, eta_b, basis=0)
        q_x, e_x = self.wcp_stats(mu_a, mu_b, eta_a, eta_b, basis=1)
        y11, e11 = self.single_photon_yield(eta_a, eta_b)
        q11 = mu_a * mu_b * np.exp(-mu_a - mu_b) * y11
        rate = q11 * (1.0 - binary_entropy(e11)) - self.ch.f_ec * q_z * binary_entropy(e_z)
        return {
            "distance_km": distance_km,
            "arm_transmittance": eta_a,
            "Q_Z": q_z, "E_Z": e_z,
            "Q_X": q_x, "E_X": e_x,
            "Y11": y11, "e11_X": e11, "Q11": q11,
            "key_rate": float(max(rate, 0.0)),
        }

    def optimize(self, distance_km: float) -> Tuple[float, Dict[str, float]]:
        """Maximizes the key rate over the two signal intensities."""
        def neg(v):
            r = self.key_rate(distance_km, v[0], v[1])["key_rate"]
            return -np.log10(r) if r > 0 else 50.0

        best = None
        for s in ([0.3, 0.3], [0.1, 0.1], [0.5, 0.5]):
            res = minimize(neg, s, bounds=[(0.01, 1.0)] * 2, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
        mu = float(best.x[0])
        return mu, self.key_rate(distance_km, best.x[0], best.x[1])


# ---------------------------------------------------------------------------
# The structural claim: what happens when Eve owns the measurement device
# ---------------------------------------------------------------------------

def bb84_blinding_attack(p_control: float, blinding_residual_qber: float = 0.005
                         ) -> Dict[str, float]:
    """Detector blinding against BB84: Eve owns Bob's detectors.

    Eve intercepts, measures in Bob's basis (she learns it from the public
    announcement in a faked-state attack), and injects a bright classical pulse
    that forces exactly the outcome she wants. She therefore learns the key bit
    on every controlled pulse, and the only error she leaves is her own
    state-preparation imperfection.

    The decisive point: in BB84 the X ("check") basis is measured by the SAME
    detectors Eve controls, so she fakes the security monitor too. There is no
    independent quantity that rises.
    """
    p = float(np.clip(p_control, 0.0, 1.0))
    return {
        "p_control": p,
        "observed_qber": blinding_residual_qber * p + 0.0 * (1 - p),
        "check_basis_error": blinding_residual_qber * p,   # also forged by Eve
        "eve_information": p,                              # bits per sifted bit
        "protocol_aborts": False,
    }


def mdi_charlie_attack(p_control: float, e_z_honest: float = 0.005,
                       e_x_honest: float = 0.030, abort_threshold: float = 0.11
                       ) -> Dict[str, float]:
    """Eve owns Charlie entirely in MDI-QKD and tries the same trick.

    Her best strategy for learning Z-basis key bits is to measure both incoming
    pulses in Z, then announce |psi-> only on the events where she saw opposite
    bits. That costs her nothing in the Z basis -- the announced events really do
    carry anticorrelated bits, so E_Z stays clean and she knows every bit.

    But measuring in Z destroys the X-basis coherence of the very same events.
    Alice and Bob prepared those X states themselves and know what they sent, so
    when they compare against Charlie's announcement the phase error on the
    intercepted fraction is 1/2 (random). Eve cannot forge this: she does not
    know what Alice and Bob prepared, and the comparison happens in their labs,
    not in hers.

    So the attack that is invisible in BB84 is loud in MDI-QKD -- and it is loud
    in the quantity that actually bounds Eve's information.
    """
    p = float(np.clip(p_control, 0.0, 1.0))
    e_z = e_z_honest                       # she keeps the Z basis clean
    e_x = (1.0 - p) * e_x_honest + p * 0.5  # phase error she cannot hide
    return {
        "p_control": p,
        "observed_qber": e_z,
        "check_basis_error": e_x,
        "eve_information": p,
        "protocol_aborts": bool(e_x > abort_threshold),
        "secure_key_rate_factor": float(max(0.0, 1.0 - binary_entropy(min(e_x, 0.5)))),
    }


def detector_attack_comparison(p_grid=None, blinding_residual_qber: float = 0.005,
                               e_z_honest: float = 0.005, e_x_honest: float = 0.030):
    """Side-by-side sweep of Eve's control fraction for BB84 vs MDI-QKD.

    Returns a list of dicts (one per control fraction) suitable for a DataFrame.
    """
    if p_grid is None:
        p_grid = np.linspace(0.0, 1.0, 21)
    rows = []
    for p in p_grid:
        bb = bb84_blinding_attack(p, blinding_residual_qber)
        md = mdi_charlie_attack(p, e_z_honest, e_x_honest)
        rows.append({
            "p_control": float(p),
            "bb84_qber": bb["observed_qber"],
            "bb84_check_error": bb["check_basis_error"],
            "bb84_eve_information": bb["eve_information"],
            "bb84_aborts": bb["protocol_aborts"],
            "mdi_qber": md["observed_qber"],
            "mdi_check_error": md["check_basis_error"],
            "mdi_eve_information": md["eve_information"],
            "mdi_aborts": md["protocol_aborts"],
            "mdi_key_rate_factor": md["secure_key_rate_factor"],
        })
    return rows
