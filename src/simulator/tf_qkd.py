"""Sending-or-Not-Sending Twin-Field QKD (SNS-TF-QKD) simulator.

Twin-field QKD (Lucamarini et al., Nature 2018) lets Alice and Bob each send weak
coherent pulses to an untrusted relay (Charlie) that interferes them on a beam
splitter. Because a successful event needs only ONE photon to survive half of the
channel, the key rate scales as sqrt(eta) instead of eta, which lets TF-QKD beat the
repeaterless PLOB bound. The sending-or-not-sending variant (Wang, Yu & Hu, PRA 98,
062323, 2018) is the protocol behind the longest fiber QKD demonstrations to date
(e.g. ~1000 km, Liu et al., PRL 130, 210801, 2023). Charlie is untrusted, so the
protocol is also measurement-device-independent: detector side channels are removed.

Protocol (per pulse pair, asymptotic, symmetric channel):
  Z (signal) window: each party independently "sends" a phase-randomized coherent
      state of intensity mu_z with probability eps, otherwise sends vacuum.
      Alice's bit = 1 if she sends; Bob's bit = 1 if he does NOT send.
      An effective event is exactly one of Charlie's two detectors clicking.
      Bit errors come from (both send) and (neither sends -> dark counts).
  X (decoy) window: each party sends a phase-randomized coherent state with
      intensity in {0, mu_1, mu_2}. Phases are announced afterwards; pairs with
      matching phases (|theta_A - theta_B| <= pi/M, or near pi) measure the phase
      error rate that bounds Eve's information.

Key rate (asymptotic, infinite data, p_Z -> 1; Yu, Hu, Jiang & Wang, Sci. Rep. 2019):
  R = 2 eps (1-eps) mu_z e^{-mu_z} s_1 [1 - H(e_1^ph)] - f S_Z H(E_Z)

All "gains" below are exact expectation values given the channel model; the
time-series simulator samples finite-block counts from them binomially.
"""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


def binary_entropy(x: float) -> float:
    x = float(np.clip(x, 1e-15, 1 - 1e-15))
    return float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))


@dataclass(frozen=True)
class ChannelParams:
    """Fiber + detector parameters (typical values from the TF-QKD literature)."""
    alpha_db_per_km: float = 0.2      # Standard SMF-28 attenuation (ULL fiber: ~0.16)
    eta_det: float = 0.6              # Charlie's SNSPD detection efficiency
    p_dark: float = 1e-8              # Dark-count probability per pulse per detector
    phase_noise_rad: float = 0.35     # Residual relative phase error after compensation (~3% X-basis error)
    phase_slices: int = 16            # M: number of phase slices used for X-window matching
    f_ec: float = 1.1                 # Error-correction inefficiency

    def arm_transmittance(self, distance_km: float) -> float:
        """Transmittance of one arm (Alice->Charlie or Bob->Charlie), detector included."""
        return self.eta_det * 10 ** (-self.alpha_db_per_km * (distance_km / 2.0) / 10.0)

    def channel_transmittance(self, distance_km: float) -> float:
        """End-to-end Alice<->Bob fiber transmittance (used for the PLOB bound)."""
        return 10 ** (-self.alpha_db_per_km * distance_km / 10.0)


@dataclass(frozen=True)
class SNSIntensities:
    eps: float = 0.25     # Sending probability in the Z window
    mu_z: float = 0.40    # Signal intensity
    mu_1: float = 0.10    # Weak decoy intensity (X window)
    mu_2: float = 0.40    # Strong decoy intensity (X window)


def plob_bound(distance_km: float, ch: ChannelParams) -> float:
    """Repeaterless secret-key capacity -log2(1 - eta) (Pirandola et al., Nat. Commun. 2017)."""
    eta = ch.channel_transmittance(distance_km)
    return float(-np.log2(1.0 - eta))


class SNSTFQKD:
    """Exact asymptotic model + finite-block sampler for SNS-TF-QKD."""

    _PHASE_GRID = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)

    def __init__(self, channel: Optional[ChannelParams] = None):
        self.ch = channel or ChannelParams()
        # Grids for averaging over the announced phase difference inside a slice and
        # the residual Gaussian phase noise (Gauss-Hermite quadrature)
        half = np.pi / self.ch.phase_slices
        self._slice_grid = np.linspace(-half, half, 65)
        gh_x, gh_w = np.polynomial.hermite_e.hermegauss(31)
        self._gh_x, self._gh_w = gh_x, gh_w / gh_w.sum()

    # ---------- Charlie's interference measurement ----------
    def _one_click_probs(self, mu_a: float, mu_b: float, eta: float, dphi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """P(only left clicks), P(only right clicks) for coherent inputs with relative phase dphi.

        Beam-splitter outputs are coherent states with mean photon numbers
        lambda_{L,R} = (a^2 + b^2 +/- 2ab cos dphi) / 2, a = sqrt(mu_a eta), b = sqrt(mu_b eta).
        """
        a, b = np.sqrt(mu_a * eta), np.sqrt(mu_b * eta)
        cross = 2.0 * a * b * np.cos(dphi)
        lam_l = (a * a + b * b + cross) / 2.0
        lam_r = (a * a + b * b - cross) / 2.0
        pd = self.ch.p_dark
        no_l = (1.0 - pd) * np.exp(-lam_l)
        no_r = (1.0 - pd) * np.exp(-lam_r)
        return (1.0 - no_l) * no_r, (1.0 - no_r) * no_l

    def gain(self, mu_a: float, mu_b: float, eta: float) -> float:
        """Effective (exactly-one-click) gain for phase-randomized inputs."""
        left, right = self._one_click_probs(mu_a, mu_b, eta, self._PHASE_GRID)
        return float(np.mean(left + right))

    def phase_matched_stats(
        self, mu: float, eta: float, phase_tamper_frac: float = 0.0, phase_noise_rad: Optional[float] = None
    ) -> Tuple[float, float]:
        """(gain, error-gain) for X-window mu-mu pairs whose announced phases match.

        Announced phase difference is uniform within +/- pi/M; the physical phase also
        carries Gaussian residual noise. With 0 difference the left detector should click;
        the pi-shifted slice is symmetric (right should click) and gives identical stats.
        A fraction `phase_tamper_frac` of pulses has its phase scrambled by Eve
        (phenomenological measure-and-resend on the phase reference).
        """
        sigma = self.ch.phase_noise_rad if phase_noise_rad is None else phase_noise_rad
        dphi = (self._slice_grid[:, None] + sigma * self._gh_x[None, :])
        w = np.broadcast_to(self._gh_w[None, :] / len(self._slice_grid), dphi.shape)
        left, right = self._one_click_probs(mu, mu, eta, dphi)
        g_ok, e_ok = float(np.sum(w * (left + right))), float(np.sum(w * right))
        if phase_tamper_frac <= 0.0:
            return g_ok, e_ok
        l_r, r_r = self._one_click_probs(mu, mu, eta, self._PHASE_GRID)
        g_rand, e_rand = float(np.mean(l_r + r_r)), float(np.mean(r_r))
        p = phase_tamper_frac
        return (1 - p) * g_ok + p * g_rand, (1 - p) * e_ok + p * e_rand

    # ---------- Protocol statistics ----------
    def statistics(
        self, distance_km: float, x: SNSIntensities, phase_tamper_frac: float = 0.0,
        phase_noise_rad: Optional[float] = None,
    ) -> Dict[str, float]:
        eta = self.ch.arm_transmittance(distance_km)
        eps, mz, m1, m2 = x.eps, x.mu_z, x.mu_1, x.mu_2

        # Z window
        s00 = self.gain(0.0, 0.0, eta)
        s_z0 = self.gain(mz, 0.0, eta)
        s_zz = self.gain(mz, mz, eta)
        n_err = (1 - eps) ** 2 * s00 + eps ** 2 * s_zz
        s_z = n_err + 2 * eps * (1 - eps) * s_z0
        e_z = n_err / s_z

        # X window decoys (phase-randomized pairs: total photon number ~ Poisson(2 mu))
        s11 = self.gain(m1, m1, eta)
        s22 = self.gain(m2, m2, eta)
        s1_l = (m2 ** 2 * np.exp(2 * m1) * s11 - m1 ** 2 * np.exp(2 * m2) * s22 - (m2 ** 2 - m1 ** 2) * s00) / (
            2 * m1 * m2 * (m2 - m1)
        )
        g_x, t_x = self.phase_matched_stats(m1, eta, phase_tamper_frac, phase_noise_rad)
        e_x = t_x / g_x
        if s1_l > 0:
            e1_ph = (t_x - np.exp(-2 * m1) * s00 / 2.0) / (2 * m1 * np.exp(-2 * m1) * s1_l)
        else:
            e1_ph = 0.5
        e1_ph = float(np.clip(e1_ph, 0.0, 0.5))

        n1 = 2 * eps * (1 - eps) * mz * np.exp(-mz) * max(s1_l, 0.0)
        rate = n1 * (1 - binary_entropy(e1_ph)) - self.ch.f_ec * s_z * binary_entropy(e_z)
        return {
            "distance_km": distance_km,
            "arm_transmittance": eta,
            "S_Z": s_z, "E_Z": e_z,
            "S_X": g_x, "E_X": e_x,
            "s1_lower": float(s1_l), "e1_phase_upper": e1_ph,
            "key_rate": float(max(rate, 0.0)),
        }

    def optimize(self, distance_km: float, x0: Optional[SNSIntensities] = None) -> Tuple[SNSIntensities, Dict[str, float]]:
        """Maximizes the key rate over (eps, mu_z, mu_1, mu_2 - mu_1) with box bounds."""
        def unpack(v):
            return SNSIntensities(eps=v[0], mu_z=v[1], mu_1=v[2], mu_2=v[2] + v[3])

        def neg_log_rate(v):
            r = self.statistics(distance_km, unpack(v))["key_rate"]
            return -np.log10(r) if r > 0 else 50.0

        bounds = [(0.01, 0.5), (0.05, 1.0), (0.005, 0.3), (0.05, 1.0)]
        # Long distances need small eps and weak decoys; large-eps starts sit on a flat
        # zero-rate plateau where the gradient vanishes, so include both regimes.
        starts = [[0.04, 0.45, 0.005, 0.06], [0.06, 0.45, 0.02, 0.2], [0.1, 0.5, 0.05, 0.4], [0.25, 0.4, 0.1, 0.3]]
        if x0 is not None:
            starts.insert(0, [x0.eps, x0.mu_z, x0.mu_1, max(x0.mu_2 - x0.mu_1, 0.05)])
        best = None
        for s in starts:
            res = minimize(neg_log_rate, s, bounds=bounds, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
        xs = unpack(best.x)
        return xs, self.statistics(distance_km, xs)


def decoy_bb84_key_rate(
    distance_km: float, ch: ChannelParams, e_d: float = 0.015, mu: Optional[float] = None, sift: float = 1.0
) -> Tuple[float, float]:
    """Asymptotic decoy-state BB84 key rate (GLLP + infinite decoys; Lo, Ma & Chen, PRL 2005).

    Point-to-point link: the full distance attenuates the pulse before Bob's detector.
    `sift` = 1.0 is efficient BB84 (biased basis choice); 0.5 is the original protocol.
    Returns (key rate per pulse, optimal mu).
    """
    eta = ch.eta_det * ch.channel_transmittance(distance_km)
    y0 = 2 * ch.p_dark

    def rate(m):
        q_mu = y0 + 1 - np.exp(-eta * m)
        e_mu = (0.5 * y0 + e_d * (1 - np.exp(-eta * m))) / q_mu
        y1 = y0 + eta - y0 * eta
        e1 = (0.5 * y0 + e_d * eta) / y1
        q1 = y1 * m * np.exp(-m)
        return sift * (q1 * (1 - binary_entropy(e1)) - ch.f_ec * q_mu * binary_entropy(e_mu))

    if mu is not None:
        return max(rate(mu), 0.0), mu
    grid = np.linspace(0.02, 1.0, 197)
    vals = np.array([rate(m) for m in grid])
    i = int(np.argmax(vals))
    return max(float(vals[i]), 0.0), float(grid[i])


@dataclass
class TFSession:
    """Per-block time series of an SNS-TF-QKD session."""
    qber_z: np.ndarray            # Z-window bit error rate per block
    phase_err_x: np.ndarray       # X-window phase-matched error rate per block
    n_x_events: np.ndarray        # Effective X-window events per block
    phase_noise: np.ndarray       # True residual phase noise per block (rad)
    block_classes: np.ndarray     # 0 benign, 2 phase-tampering attack active
    p_eve: float
    attack_start_block: int


class SNSTimeSeriesSimulator:
    """Finite-block SNS-TF-QKD sessions with drifting phase noise and optional phase tampering.

    Residual phase noise (fiber length fluctuations left after active phase compensation)
    drifts as an AR(1) process in log space around ChannelParams.phase_noise_rad.
    From attack_start_block on, Eve scrambles the phase of a fraction p_eve of pulses.
    This leaves Z-window statistics unchanged (Z pulses are phase-randomized anyway)
    but raises the X-window phase error: the attack is invisible to a QBER monitor.
    """

    def __init__(
        self, model: SNSTFQKD, distance_km: float, intensities: SNSIntensities,
        z_pairs_per_block: float = 2e9, x_pairs_per_block: float = 2e9,
        drift_std: float = 0.15, drift_phi: float = 0.8, random_state: Optional[int] = None,
    ):
        self.model = model
        self.distance_km = distance_km
        self.x = intensities
        self.nz = int(float(z_pairs_per_block))   # float() also accepts YAML strings like "2.0e7"
        self.nx = int(float(x_pairs_per_block))
        self.drift_std = float(drift_std)
        self.drift_phi = float(drift_phi)
        self.rng = np.random.default_rng(random_state)

    def simulate_session(self, attack: bool, p_eve: float = 0.0, blocks: int = 200, attack_start_block: int = 101) -> TFSession:
        qz, ex, nx, noise, cls = (np.zeros(blocks) for _ in range(5))
        z = 0.0
        sigma0 = self.model.ch.phase_noise_rad
        for b in range(blocks):
            z = self.drift_phi * z + np.sqrt(1 - self.drift_phi ** 2) * self.rng.normal(0.0, self.drift_std)
            sigma_t = sigma0 * np.exp(z - 0.5 * self.drift_std ** 2)
            active = attack and (b + 1) >= attack_start_block
            st = self.model.statistics(self.distance_km, self.x, p_eve if active else 0.0, sigma_t)
            n_z = self.rng.binomial(self.nz, st["S_Z"])
            qz[b] = self.rng.binomial(n_z, st["E_Z"]) / max(n_z, 1)
            n_x = self.rng.binomial(self.nx, st["S_X"])
            ex[b] = self.rng.binomial(n_x, st["E_X"]) / max(n_x, 1)
            nx[b], noise[b], cls[b] = n_x, sigma_t, 2 if active else 0
        return TFSession(qz, ex, nx.astype(int), noise, cls.astype(int), p_eve if attack else 0.0, attack_start_block)
