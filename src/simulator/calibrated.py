"""Physically calibrated BB84 simulator (weak coherent pulses, count-level sampling).

Why a second engine
-------------------
`bb84_engine.BB84Simulator` draws every pulse individually with ~1,000 pulses per
block. That is only affordable because it treats each pulse as a perfect single
photon that is detected with probability T*eta. Real weak-coherent-pulse (WCP)
systems detect a fraction of a percent of pulses, so a block needs ~10^5-10^6
pulses for a usable sifted key. This engine therefore works at the level of
*pulse populations*: for every (intensity, Alice basis, Bob basis) cell it computes
the exact click and error probabilities from photon statistics, then samples the
counts binomially. That is statistically identical to pulse-by-pulse simulation
for i.i.d. pulses, and ~10^4 times faster.

Physics included (all parameters in config/physics.yaml with sources)
  * Poisson photon number, mean mu, for every class (fixes audit B-2); optional
    signal/decoy/vacuum intensities.
  * Fibre loss 10^(-alpha L / 10) or free-space log-normal fading with AR(1)
    temporal correlation.
  * Four detectors (Z0, Z1, X0, X1), per-detector efficiency, dark counts that
    produce random bits (fixes audit B-1), double clicks squashed to a random bit,
    optional afterpulsing and dead time.
  * Optical misalignment e_d with benign drift, benign loss steps and detector
    efficiency drift in EVERY session (fixes audit L-1).
  * Per-block QBER Wilson confidence interval.

Attack models (each a deliberate simplification, see docs/ATTACK_MODELS.md)
  intercept_resend, pns, detector_blinding, time_shift, trojan_horse, beam_splitting.

Coherent-state outputs are exact: a coherent pulse of mean lambda split over
detectors gives independent Poisson counts, so a click on detector i has
probability 1 - (1 - p_d) exp(-lambda_i). Fock-state outputs (PNS forwarding) use
the multinomial no-click probabilities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.evaluation.finite_size import wilson_interval
from src.simulator.bb84_engine import SessionSequence

CLASSES: List[Tuple[int, str, str]] = [
    (0, "normal", "Normal Channel"),
    (1, "intercept_resend", "Fractional Intercept-Resend"),
    (2, "pns", "Photon Number Splitting"),
    (3, "detector_blinding", "Detector Blinding (Fake State)"),
    (4, "time_shift", "Time-Shift"),
    (5, "trojan_horse", "Trojan Horse (simplified)"),
    (6, "beam_splitting", "Beam Splitting"),
]
CLASS_KEYS = [c[1] for c in CLASSES]
CLASS_NAMES = [c[2] for c in CLASSES]


# ---------------------------------------------------------------------------
# Click statistics
# ---------------------------------------------------------------------------
def coherent_clicks(lam_c, lam_w, pd_c, pd_w):
    """Correct/wrong detector receive coherent light of mean lam_c / lam_w.

    Returns (P(any click), P(click AND wrong bit)); a double click counts as a
    random bit (squashing), i.e. half an error.
    """
    c = 1.0 - (1.0 - pd_c) * np.exp(-lam_c)
    w = 1.0 - (1.0 - pd_w) * np.exp(-lam_w)
    det = 1.0 - (1.0 - c) * (1.0 - w)
    err = w * (1.0 - c) + 0.5 * c * w
    return det, err


def fock_clicks(m, q_c, q_w, pd_c, pd_w):
    """m photons, each detected on the correct / wrong detector with prob q_c / q_w."""
    m = np.asarray(m, dtype=np.float64)
    none = (1.0 - pd_c) * (1.0 - pd_w) * np.power(np.clip(1.0 - q_c - q_w, 0.0, 1.0), m)
    c_silent = (1.0 - pd_c) * np.power(np.clip(1.0 - q_c, 0.0, 1.0), m)
    w_silent = (1.0 - pd_w) * np.power(np.clip(1.0 - q_w, 0.0, 1.0), m)
    only_c = w_silent - none
    only_w = c_silent - none
    both = 1.0 - c_silent - w_silent + none
    return 1.0 - none, only_w + 0.5 * both


def poisson_pmf(mu, n_max):
    n = np.arange(n_max + 1, dtype=np.float64)
    from scipy.special import gammaln
    return np.exp(-mu + n * np.log(max(mu, 1e-300)) - gammaln(n + 1)) if mu > 0 else (n == 0).astype(float)


# ---------------------------------------------------------------------------
# Per-session physical parameters
# ---------------------------------------------------------------------------
@dataclass
class SessionPhysics:
    channel_type: str
    distance_km: float
    loss_db: float
    detector_type: str
    eta: np.ndarray                 # shape (2 bases, 2 bits)
    p_dark: float
    afterpulse: float
    dead_time_s: float
    e_d0: float
    intensities: np.ndarray
    intensity_probs: np.ndarray
    scint_index: float = 0.0
    fading_phi: float = 0.0
    loss_step_block: int = -1
    loss_step_db: float = 0.0
    attack: str = "normal"
    intensity: float = 0.0
    attack_params: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "channel_type": self.channel_type, "distance_km": self.distance_km, "loss_db": self.loss_db,
            "detector_type": self.detector_type, "eta_mean": float(self.eta.mean()), "p_dark": self.p_dark,
            "afterpulse": self.afterpulse, "e_d0": self.e_d0, "mu": float(self.intensities[0]),
            "scint_index": self.scint_index, "loss_step_block": self.loss_step_block,
            "loss_step_db": self.loss_step_db, "attack": self.attack, "intensity": self.intensity,
            **{f"atk_{k}": v for k, v in self.attack_params.items()},
        }


class CalibratedBB84Simulator:
    """Count-level WCP BB84 simulator driven by config/physics.yaml (`calibrated_bb84`)."""

    def __init__(self, physics: Dict[str, Any], random_state: Optional[int] = None):
        self.p = physics
        self.seed_seq = np.random.SeedSequence(random_state)
        self.rng = np.random.default_rng(self.seed_seq)

    # ---------------- sampling helpers ----------------
    @staticmethod
    def _u(rng, rng_pair):
        lo, hi = rng_pair
        return float(rng.uniform(lo, hi)) if hi > lo else float(lo)

    def draw_session_physics(self, rng, attack: str, channel_type: str = "fiber",
                             heldout: bool = False, blocks: int = 150) -> SessionPhysics:
        """Channel and detector values come from ONE distribution for every class.

        The nuisance draws happen in a fixed order before anything attack-specific,
        so a benign and an attacked session with the same seed share the same channel.
        """
        p = self.p
        src, ch, det, opt, dyn = p["source"], p["channel"], p["detector"], p["optics"], p["benign_dynamics"]

        if channel_type == "fiber":
            fc = ch["fiber"]
            dist = self._u(rng, p["heldout"]["distance_km"] if heldout else fc["distance_km"])
            loss_db = fc["attenuation_db_per_km"] * dist + self._u(rng, fc["extra_loss_db"])
            scint, phi, bg = 0.0, 0.0, 0.0
        elif channel_type == "free_space":
            fs = ch["free_space"]
            dist = self._u(rng, fs["distance_km"])
            loss_db = self._u(rng, fs["geometric_loss_db"]) + self._u(rng, fs["atmospheric_attenuation_db_per_km"]) * dist
            scint, phi = self._u(rng, fs["scintillation_index"]), float(fs["fading_ar1_phi"])
            bg = 10.0 ** self._u(rng, fs["log_background_per_gate"])
        else:
            raise ValueError(f"unknown channel_type {channel_type}")

        dtype = str(det["types"][int(rng.integers(len(det["types"])))])
        dcfg = det[dtype]
        eta0 = self._u(rng, dcfg["efficiency"])
        mism = self._u(rng, det["per_basis_efficiency_mismatch"])
        eta = eta0 * (1.0 + mism * (rng.random((2, 2)) - 0.5))
        p_dark = 10.0 ** self._u(rng, dcfg["log_dark_per_gate"]) + bg
        ap = self._u(rng, dcfg["afterpulse_prob"]) if det.get("enable_afterpulsing", False) else 0.0
        dt = float(dcfg["dead_time_s"]) if det.get("enable_dead_time", False) else 0.0
        e_d0 = self._u(rng, opt["misalignment_error"])

        if src["decoy"]["enabled"]:
            mus = np.asarray(src["decoy"]["intensities"], dtype=float)
            probs = np.asarray(src["decoy"]["probabilities"], dtype=float)
        else:
            mus, probs = np.array([src["mean_photon_number"]], dtype=float), np.array([1.0])

        step_draw = rng.random()
        step_block = int(rng.integers(1, blocks)) if step_draw < dyn["loss_step_prob"] else -1
        step_db = self._u(rng, dyn["loss_step_db"]) if step_block > 0 else 0.0

        sp = SessionPhysics(channel_type, dist, loss_db, dtype, eta, p_dark, ap, dt, e_d0,
                            mus, probs / probs.sum(), scint, phi, step_block, step_db)

        if attack != "normal":
            a = p["attacks"][attack]
            sp.attack = attack
            sp.intensity = self._u(rng, a["heldout_intensity"] if heldout else a["intensity"])
            if attack == "pns":
                sp.attack_params["gain_mismatch"] = self._u(rng, a["gain_mismatch"]) * rng.choice([-1, 1])
            elif attack == "detector_blinding":
                sp.attack_params["residual_qber"] = self._u(rng, a["residual_qber"])
                sp.attack_params["gain_mismatch"] = self._u(rng, a["trigger_gain_mismatch"]) * rng.choice([-1, 1])
            elif attack == "time_shift":
                sp.attack_params["efficiency_ratio"] = self._u(rng, a["efficiency_ratio"])
            elif attack == "trojan_horse":
                sp.attack_params["forward_leak_ratio"] = 10.0 ** self._u(
                    rng, (np.log10(a["forward_leak_ratio"][0]), np.log10(a["forward_leak_ratio"][1])))
        return sp

    # ---------------- block-level dynamics ----------------
    def _block_dynamics(self, rng, sp: SessionPhysics, blocks: int):
        dyn = self.p["benign_dynamics"]
        t = np.full(blocks, 10.0 ** (-sp.loss_db / 10.0))
        if sp.loss_step_block > 0:
            t[sp.loss_step_block:] *= 10.0 ** (-sp.loss_step_db / 10.0)
        if sp.channel_type == "free_space" and sp.scint_index > 0:
            s2 = np.log1p(sp.scint_index)                  # log-intensity variance
            z = np.empty(blocks)
            z[0] = rng.normal(0.0, np.sqrt(s2))
            innov = rng.normal(0.0, np.sqrt(s2 * (1 - sp.fading_phi ** 2)), blocks)
            for i in range(1, blocks):
                z[i] = sp.fading_phi * z[i - 1] + innov[i]
            t *= np.exp(z - s2 / 2.0)                      # unit-mean log-normal fading
            t = np.clip(t, 0.0, 1.0)
        ed = np.empty(blocks)
        x = 0.0
        innov = rng.normal(0.0, dyn["misalignment_drift_std"], blocks)
        for i in range(blocks):
            x = dyn["misalignment_drift_phi"] * x + innov[i]
            ed[i] = x
        ed = np.clip(sp.e_d0 + ed, 0.0, 0.08)
        eff_drift = np.exp(np.cumsum(rng.normal(0.0, dyn["efficiency_drift_std"], blocks)))
        eff_drift = np.clip(eff_drift, 0.7, 1.3)
        return t, ed, eff_drift

    # ---------------- per-block cell statistics ----------------
    def _cell_stats(self, sp: SessionPhysics, t: float, ed: float, eta: np.ndarray,
                    active: bool) -> Dict[str, np.ndarray]:
        """Probabilities for every (intensity k, Bob basis b, matching?) cell.

        Returns arrays shaped (K, 2): det_match, err_match, det_mis, plus eve_info (K, 2).
        """
        K = len(sp.intensities)
        pd = sp.p_dark
        det_m = np.zeros((K, 2)); err_m = np.zeros((K, 2)); det_x = np.zeros((K, 2)); info = np.zeros((K, 2))
        attack = sp.attack if active else "normal"
        f = sp.intensity
        for b in range(2):
            e0, e1 = eta[b]
            for k, mu in enumerate(sp.intensities):
                lam = mu * t
                # ----- honest population -----
                hd0, he0 = coherent_clicks(lam * e0 * (1 - ed), lam * e1 * ed, pd, pd)   # Alice bit 0
                hd1, he1 = coherent_clicks(lam * e1 * (1 - ed), lam * e0 * ed, pd, pd)   # Alice bit 1
                hon_d, hon_e = 0.5 * (hd0 + hd1), 0.5 * (he0 + he1)
                hon_x, _ = coherent_clicks(lam * e0 / 2, lam * e1 / 2, pd, pd)
                d, e, dx, inf = hon_d, hon_e, hon_x, 0.0

                if attack == "intercept_resend":
                    d, e, dx, inf = self._ir(sp, mu, t, ed, e0, e1, pd, hon_d, f)
                elif attack == "pns":
                    d, e, dx, inf = self._pns(sp, mu, t, ed, e0, e1, pd, hon_d, hon_e, hon_x, f, b)
                elif attack == "detector_blinding":
                    d, e, dx, inf = self._blinding(sp, mu, pd, hon_d, hon_e, hon_x, f, b)
                elif attack == "time_shift":
                    # Simplified: two-level detector efficiency vs arrival time; Eve shifts
                    # early or late at random and never measures the photon.
                    r = sp.attack_params["efficiency_ratio"]
                    ts = []
                    for (s0, s1) in ((1.0, r), (r, 1.0)):            # early / late shift
                        a0, ae0 = coherent_clicks(lam * e0 * s0 * (1 - ed), lam * e1 * s1 * ed, pd, pd)
                        a1, ae1 = coherent_clicks(lam * e1 * s1 * (1 - ed), lam * e0 * s0 * ed, pd, pd)
                        ax, _ = coherent_clicks(lam * e0 * s0 / 2, lam * e1 * s1 / 2, pd, pd)
                        ts.append((0.5 * (a0 + a1), 0.5 * (ae0 + ae1), ax))
                    sd = 0.5 * (ts[0][0] + ts[1][0]); se = 0.5 * (ts[0][1] + ts[1][1]); sx = 0.5 * (ts[0][2] + ts[1][2])
                    d, e, dx = (1 - f) * hon_d + f * sd, (1 - f) * hon_e + f * se, (1 - f) * hon_x + f * sx
                    inf = f * (1 - r) / (1 + r)
                elif attack == "trojan_horse":
                    # Simplified: Bob only sees Eve's leaked light as extra noise clicks; the
                    # real signature is at Alice's watchdog (recorded as alice_monitor).
                    p_noise = sp.attack_params["forward_leak_ratio"] * f
                    pdn = pd + p_noise
                    d0, ee0 = coherent_clicks(lam * e0 * (1 - ed), lam * e1 * ed, pdn, pdn)
                    d1, ee1 = coherent_clicks(lam * e1 * (1 - ed), lam * e0 * ed, pdn, pdn)
                    d, e = 0.5 * (d0 + d1), 0.5 * (ee0 + ee1)
                    dx, _ = coherent_clicks(lam * e0 / 2, lam * e1 / 2, pdn, pdn)
                    inf = 1.0 - np.exp(-f)
                elif attack == "beam_splitting":
                    # Simplified: passive tap; Eve compensates the loss with a better link.
                    beta = f
                    comp = self.p["attacks"]["beam_splitting"].get("loss_compensation", True)
                    lam_b = mu * min(t, 1.0 - beta) if comp else mu * (1.0 - beta) * t
                    d0, ee0 = coherent_clicks(lam_b * e0 * (1 - ed), lam_b * e1 * ed, pd, pd)
                    d1, ee1 = coherent_clicks(lam_b * e1 * (1 - ed), lam_b * e0 * ed, pd, pd)
                    d, e = 0.5 * (d0 + d1), 0.5 * (ee0 + ee1)
                    dx, _ = coherent_clicks(lam_b * e0 / 2, lam_b * e1 / 2, pd, pd)
                    inf = 1.0 - np.exp(-mu * beta)
                det_m[k, b], err_m[k, b], det_x[k, b], info[k, b] = d, e, dx, inf
        return {"det_match": det_m, "err_match": err_m, "det_mis": det_x, "eve_info": info}

    def _ir(self, sp, mu, t, ed, e0, e1, pd, hon_d, f):
        """Intercept-resend on a fraction f of pulses, with optional gain matching.

        Simplifications: Eve's detector is perfect (she misses only vacuum pulses), she
        resends one coherent intensity for every pulse, and there is no timing signature.
        """
        p_click_eve = 1.0 - np.exp(-mu)
        dark_d, dark_e = coherent_clicks(0.0, 0.0, pd, pd)
        eta_bar = 0.5 * (e0 + e1)
        if self.p["attacks"]["intercept_resend"].get("gain_matching", True):
            # Eve cannot tell decoys from signals, so she picks ONE resend intensity that
            # reproduces Bob's honest click rate on the signal pulses and uses it for all.
            sig = sp.intensities[0]
            sig_hon = 0.5 * (coherent_clicks(sig * t * e0 * (1 - ed), sig * t * e1 * ed, pd, pd)[0]
                             + coherent_clicks(sig * t * e1 * (1 - ed), sig * t * e0 * ed, pd, pd)[0])
            p_sig = 1.0 - np.exp(-sig)
            target = np.clip((sig_hon - np.exp(-sig) * dark_d) / max(p_sig, 1e-12), 0.0, 0.999999)
            lam_e = max(-np.log(max(1.0 - target, 1e-12) / (1.0 - pd) ** 2) / max(eta_bar, 1e-12), 0.0)
        else:
            lam_e = mu * t
        # Eve basis = Alice basis: Bob receives Eve's correct bit
        a0, ae0 = coherent_clicks(lam_e * e0 * (1 - ed), lam_e * e1 * ed, pd, pd)
        a1, ae1 = coherent_clicks(lam_e * e1 * (1 - ed), lam_e * e0 * ed, pd, pd)
        same_d, same_e = 0.5 * (a0 + a1), 0.5 * (ae0 + ae1)
        # Eve basis != Alice basis: Bob's outcome is random w.r.t. Alice's bit
        diff_d, diff_e = coherent_clicks(lam_e * e0 / 2, lam_e * e1 / 2, pd, pd)
        diff_e = 0.5 * diff_d
        eve_d = p_click_eve * 0.5 * (same_d + diff_d) + np.exp(-mu) * dark_d
        eve_e = p_click_eve * 0.5 * (same_e + diff_e) + np.exp(-mu) * dark_e
        eve_x = p_click_eve * diff_d + np.exp(-mu) * dark_d
        hon_e = 0.5 * (coherent_clicks(mu * t * e0 * (1 - ed), mu * t * e1 * ed, pd, pd)[1]
                       + coherent_clicks(mu * t * e1 * (1 - ed), mu * t * e0 * ed, pd, pd)[1])
        hon_x = coherent_clicks(mu * t * e0 / 2, mu * t * e1 / 2, pd, pd)[0]
        info = f * p_click_eve * 0.5
        return (1 - f) * hon_d + f * eve_d, (1 - f) * hon_e + f * eve_e, (1 - f) * hon_x + f * eve_x, info

    def _pns(self, sp, mu, t, ed, e0, e1, pd, hon_d, hon_e, hon_x, f, b):
        """QND photon-number splitting with lossless forwarding and yield matching to the signal.

        Simplifications: ideal QND photon counting and quantum memory; Eve forwards
        n-1 photons with no loss; she tunes forwarding on the signal intensity only,
        with a per-session relative yield error `gain_mismatch`.
        """
        n_max = 12
        sig = sp.intensities[0]
        eta_bar = 0.5 * (e0 + e1)
        qc, qw = eta_bar * (1 - ed), eta_bar * ed
        m = np.arange(n_max + 1)
        fd, fe = fock_clicks(m, qc, qw, pd, pd)                  # m photons arriving at Bob
        fdx, _ = fock_clicks(m, eta_bar / 2, eta_bar / 2, pd, pd)
        dark_d, dark_e = fd[0], fe[0]

        def eve_stats(intensity, s_single, r_multi):
            pn = poisson_pmf(intensity, n_max)
            y = np.zeros(n_max + 1); ye = np.zeros(n_max + 1); yx = np.zeros(n_max + 1)
            y[0], ye[0], yx[0] = dark_d, dark_e, fdx[0]
            y[1] = s_single * fd[1] + (1 - s_single) * dark_d
            ye[1] = s_single * fe[1] + (1 - s_single) * dark_e
            yx[1] = s_single * fdx[1] + (1 - s_single) * fdx[0]
            y[2:] = r_multi * fd[1:-1] + (1 - r_multi) * dark_d          # keeps one photon, forwards n-1
            ye[2:] = r_multi * fe[1:-1] + (1 - r_multi) * dark_e
            yx[2:] = r_multi * fdx[1:-1] + (1 - r_multi) * fdx[0]
            multi_frac = float(np.sum(pn[2:] * (y[2:] - dark_d)))
            return float(pn @ y), float(pn @ ye), float(pn @ yx), multi_frac

        # Eve tunes (s, r) on the signal intensity only: she cannot tell decoys apart.
        sig_hon = hon_d if mu == sig else 0.5 * (
            coherent_clicks(sig * t * e0 * (1 - ed), sig * t * e1 * ed, pd, pd)[0]
            + coherent_clicks(sig * t * e1 * (1 - ed), sig * t * e0 * ed, pd, pd)[0])
        target = sig_hon * (1.0 + sp.attack_params.get("gain_mismatch", 0.0))
        g_multi, *_ = eve_stats(sig, 0.0, 1.0)
        if g_multi >= target:
            g0, *_ = eve_stats(sig, 0.0, 0.0)
            s, r = 0.0, float(np.clip((target - g0) / max(g_multi - g0, 1e-15), 0.0, 1.0))
        else:
            g1, *_ = eve_stats(sig, 1.0, 1.0)
            s, r = float(np.clip((target - g_multi) / max(g1 - g_multi, 1e-15), 0.0, 1.0)), 1.0
        d, e, dx, multi = eve_stats(mu, s, r)
        info = f * multi / max(d, 1e-15)
        return (1 - f) * hon_d + f * d, (1 - f) * hon_e + f * e, (1 - f) * hon_x + f * dx, info

    def _blinding(self, sp, mu, pd, hon_d, hon_e, hon_x, f, b):
        """Blinding + faked states: Bob clicks only when his basis equals Eve's.

        Simplifications: blinding is perfect while active (no dark counts, no
        afterpulses), there is no photocurrent or temperature telemetry, and the trigger
        click probability is tuned to Bob's honest gain with a relative error.
        """
        eps = sp.attack_params["residual_qber"]
        p_eve = 1.0 - np.exp(-mu)
        sig = sp.intensities[0]
        xi = np.clip(hon_d / max(0.5 * (1.0 - np.exp(-sig)), 1e-12), 0.0, 1.0) * (1.0 + sp.attack_params["gain_mismatch"])
        xi = float(np.clip(xi, 0.0, 1.0))
        d = p_eve * 0.5 * xi                      # blinded detectors: no dark counts
        e = d * eps
        dx = p_eve * 0.5 * xi
        return (1 - f) * hon_d + f * d, (1 - f) * hon_e + f * e, (1 - f) * hon_x + f * dx, f * p_eve * 0.5

    # ---------------- session ----------------
    def simulate_session(self, session_id: int, target_attack_class: int, channel_type: str = "fiber",
                         blocks_per_session: int = 150, attack_start_block: int = 76,
                         heldout: bool = False, session_seed: Optional[int] = None,
                         physics: Optional[SessionPhysics] = None) -> SessionSequence:
        rng = np.random.default_rng(session_seed) if session_seed is not None else self.rng
        attack = CLASS_KEYS[target_attack_class]
        sp = physics or self.draw_session_physics(rng, attack, channel_type, heldout, blocks_per_session)
        B = blocks_per_session
        src = self.p["source"]
        N = int(src["pulses_per_block"])
        pz = float(src["basis_prob_z"])
        conf = float(self.p["finite_size"]["confidence"])
        cap = int(self.p["finite_size"]["gap_sequence_cap"])
        f_rep = float(src["repetition_rate_hz"])

        t_arr, ed_arr, eff_drift = self._block_dynamics(rng, sp, B)
        K = len(sp.intensities)
        # cells: (k, alice basis, bob basis)
        p_basis = np.array([pz, 1 - pz])
        cell_p = (sp.intensity_probs[:, None, None] * p_basis[None, :, None] * p_basis[None, None, :]).ravel()

        out = {k: np.zeros(B) for k in (
            "qber", "qber_z", "qber_x", "sift", "sift_z", "sift_x", "err", "det", "eff", "gm", "gs", "gc", "cl",
            "lo", "hi", "info", "t", "ed", "exp_dark", "monitor")}
        q_k = np.zeros((B, K)); e_k = np.zeros((B, K))
        classes = np.zeros(B, dtype=np.int32)

        for i in range(B):
            active = (attack != "normal") and (i + 1 >= attack_start_block)
            classes[i] = target_attack_class if active else 0
            eta = sp.eta * eff_drift[i]
            st = self._cell_stats(sp, t_arr[i], ed_arr[i], eta, active)
            # Afterpulsing raises the effective dark probability in proportion to the click rate
            if sp.afterpulse > 0:
                mean_click = float(np.mean(st["det_match"]))
                sp_eff = SessionPhysics(**{**sp.__dict__, "p_dark": sp.p_dark + sp.afterpulse * mean_click / 2})
                st = self._cell_stats(sp_eff, t_arr[i], ed_arr[i], eta, active)
            # Dead time thins clicks at high count rates (non-paralysable approximation)
            thin = 1.0
            if sp.dead_time_s > 0:
                rate = float(np.mean(st["det_match"])) * f_rep
                thin = 1.0 / (1.0 + rate * sp.dead_time_s)

            n_cells = rng.multinomial(N, cell_p).reshape(K, 2, 2)
            sift_err = np.zeros((K, 2), dtype=np.int64); sift_n = np.zeros((K, 2), dtype=np.int64)
            det_total = np.zeros(K, dtype=np.int64); pulses_k = n_cells.sum(axis=(1, 2))
            for k in range(K):
                for a in range(2):
                    for bb in range(2):
                        n = int(n_cells[k, a, bb])
                        if a == bb:
                            pdet = float(np.clip(st["det_match"][k, bb] * thin, 0, 1))
                            nd = int(rng.binomial(n, pdet))
                            pe = float(np.clip(st["err_match"][k, bb] / max(st["det_match"][k, bb], 1e-300), 0, 1))
                            ne = int(rng.binomial(nd, pe))
                            sift_n[k, bb] += nd; sift_err[k, bb] += ne
                        else:
                            nd = int(rng.binomial(n, float(np.clip(st["det_mis"][k, bb] * thin, 0, 1))))
                        det_total[k] += nd

            q_k[i] = det_total / np.maximum(pulses_k, 1)
            e_k[i] = sift_err.sum(axis=1) / np.maximum(sift_n.sum(axis=1), 1)
            nz, nx = int(sift_n[0, 0]), int(sift_n[0, 1])
            ez, ex = int(sift_err[0, 0]), int(sift_err[0, 1])
            ns, es = nz + nx, ez + ex
            out["sift"][i], out["sift_z"][i], out["sift_x"][i], out["err"][i] = ns, nz, nx, es
            out["qber"][i] = es / ns if ns else 0.0
            out["qber_z"][i] = ez / nz if nz else 0.0
            out["qber_x"][i] = ex / nx if nx else 0.0
            out["det"][i] = det_total[0]
            out["eff"][i] = det_total[0] / max(pulses_k[0], 1)
            out["exp_dark"][i] = 2.0 * sp.p_dark * pulses_k[0] / max(det_total[0], 1)
            out["info"][i] = float(np.mean(st["eve_info"][0]))
            out["t"][i], out["ed"][i] = t_arr[i], ed_arr[i]
            out["monitor"][i] = sp.intensity if (active and attack == "trojan_horse") else 0.0

            # Error-gap statistics: errors are exchangeable within a block, so their
            # positions are a uniform random subset of the sifted sequence.
            n_seq = min(ns, cap)
            e_seq = int(rng.hypergeometric(es, ns - es, n_seq)) if ns > 0 and n_seq > 0 else 0
            if e_seq >= 2:
                pos = np.sort(rng.choice(n_seq, e_seq, replace=False))
                gaps = np.diff(pos)
                gm, gsd = float(gaps.mean()), float(gaps.std())
                out["gm"][i], out["gs"][i], out["gc"][i] = gm, gsd, gsd / (gm + 1e-9)
                out["cl"][i] = float(np.sum(gaps == 1)) / e_seq
            else:
                out["gm"][i] = float(n_seq) if e_seq == 0 else n_seq / 2.0

        lo, hi = wilson_interval(out["err"], out["sift"], conf)
        seq = SessionSequence(
            session_id=session_id, channel_type=channel_type, attack_class=target_attack_class,
            p_eve=sp.intensity, attack_start_block=attack_start_block,
            block_qbers=out["qber"], block_sifted_lens=out["sift"].astype(np.int64),
            block_channel_noises=out["ed"], block_classes=classes,
            block_qbers_z=out["qber_z"], block_qbers_x=out["qber_x"],
            block_sifted_z=out["sift_z"], block_sifted_x=out["sift_x"],
            block_raw_photons=np.full(B, float(N)), block_detection_counts=out["det"],
            block_transmission_effs=out["eff"], block_error_gap_means=out["gm"],
            block_error_gap_stds=out["gs"], block_error_gap_cvs=out["gc"], block_error_clusterings=out["cl"],
        )
        seq.block_errors = out["err"]
        seq.block_qber_ci_low, seq.block_qber_ci_high = lo, hi
        seq.block_expected_dark_ratio = out["exp_dark"]
        seq.block_eve_info = out["info"]
        seq.block_transmittance = out["t"]
        seq.block_alice_monitor = out["monitor"]
        seq.block_intensity_gains = q_k
        seq.block_intensity_qbers = e_k
        seq.intensities = sp.intensities.copy()
        seq.physics = sp
        return seq
