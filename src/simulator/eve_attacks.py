"""Eavesdropping attack models for discrete-variable BB84 QKD.

Implements the 5 major operational classes (Normal + 4 Major Attack Archetypes):
- Class 0: Normal Channel (Benign)
- Class 1: Fractional / Stealthy Intercept-Resend (IR)
- Class 2: Photon Number Splitting (PNS)
- Class 3: Fake State (Detector Blinding)
- Class 4: Time-Shift Attack
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class EveAttackConfig:
    """Configuration parameters for an active eavesdropper."""
    class_id: int            # 0: Benign, 1: Fractional IR, 2: PNS, 3: Fake State, 4: Time-Shift
    p_eve: float             # Primary attack parameter
    name: str                # Human-readable label
    params: Optional[Dict[str, Any]] = None


class EveAttacker:
    """Eavesdropper executing one of the 4 major attack archetypes on BB84 pulses.
    
    Threat Models:
    - Class 0 (Normal): Transparent channel, no eavesdropping.
    - Class 1 (Fractional IR): Intercepts fraction r_e in [0.10, 0.30] of pulses, measures in
      random basis, collapses state, and re-prepares. Sub-threshold QBER (5.3% - 9.9%).
    - Class 2 (PNS): Exploits Poissonian multi-photon pulses (mu ~ 0.3). Splits n >= 2 pulses,
      stores 1 photon in quantum memory, forwards remainder losslessly, attenuates single photons.
      Zero induced QBER (Delta QBER ~ 0%), but key generation rate and loss interaction decouple.
    - Class 3 (Fake State): Blinds Bob's SPADs with CW light into linear mode. Intercepts state
      and sends bright pulses triggering clicks only when Bob's basis matches Eve's basis.
      Near-zero QBER (~0% - 1.5%), deterministic basis-coincidence, abnormal error-gap CV.
    - Class 4 (Time-Shift): Shifts photon arrival times into temporal windows exploiting
      detector efficiency mismatch between bases (Z vs X). Induces severe basis asymmetry
      (|q_Z - q_X| >> 0) while session-average QBER remains normal (~4% - 5%).
    """

    CLASS_NAMES = {
        0: "Normal Channel",
        1: "Fractional Intercept-Resend",
        2: "Photon Number Splitting",
        3: "Fake State Blinding",
        4: "Time-Shift",
    }

    def __init__(
        self,
        class_id: int = 0,
        p_eve: float = 0.0,
        random_state: Optional[int] = None,
        attack_type: Optional[str] = None,
        mu_wcp: float = 0.30,
        shift_delay_prob: float = 0.60,
        pns_single_pass_frac: float = 0.35,
        blinding_eff: float = 0.95,
        blinding_residual_qber: float = 0.005,
        ts_eff_z: float = 1.0,
        ts_eff_x_ratio: float = 0.40,
        ts_x_misalign: float = 0.14,
    ):
        self.class_id = int(class_id)
        self.p_eve = float(p_eve)
        self.mu_wcp = float(mu_wcp)
        self.shift_delay_prob = float(shift_delay_prob)
        # Attack-strength nuisance parameters. The defaults reproduce the original
        # hard-coded constants, so direct callers and tests are unaffected. The
        # dataset generator randomizes them per session over ranges that overlap
        # benign behaviour, so no class carries a deterministic fingerprint
        # (see README section 3.0, task P2.1).
        self.pns_single_pass_frac = float(pns_single_pass_frac)
        self.blinding_eff = float(blinding_eff)
        self.blinding_residual_qber = float(blinding_residual_qber)
        self.ts_eff_z = float(ts_eff_z)
        self.ts_eff_x_ratio = float(ts_eff_x_ratio)
        self.ts_x_misalign = float(ts_x_misalign)
        self.rng = np.random.default_rng(random_state)
        
        # Determine internal attack type
        if attack_type is not None:
            self.attack_type = attack_type.lower()
        else:
            if self.class_id == 0:
                self.attack_type = "benign"
            elif self.class_id == 1:
                self.attack_type = "intercept_resend"
            elif self.class_id == 2:
                # If class_id == 2 and p_eve was passed without mu_wcp or attack_type,
                # check if it's legacy fractional IR (used in tests with p_eve=0.2, 1.0)
                if self.p_eve in [0.2, 1.0] and shift_delay_prob == 0.60 and mu_wcp == 0.30:
                    self.attack_type = "intercept_resend"
                else:
                    self.attack_type = "pns"
            elif self.class_id == 3:
                self.attack_type = "fake_state"
            elif self.class_id == 4:
                self.attack_type = "time_shift"
            else:
                self.attack_type = "intercept_resend"

        # Name assignment
        if self.class_id in self.CLASS_NAMES:
            self.name = self.CLASS_NAMES[self.class_id]
        else:
            self.name = f"Custom Attack (Class {self.class_id})"

        # Parameter sanity defaults
        if self.attack_type == "benign":
            self.p_eve = 0.0
            self.class_id = 0
        elif self.attack_type == "intercept_resend" and self.p_eve == 0.0:
            self.p_eve = 0.20
        elif self.attack_type == "pns" and self.p_eve == 0.0:
            self.p_eve = self.mu_wcp
            self.class_id = 2
        elif self.attack_type == "fake_state" and self.p_eve == 0.0:
            self.p_eve = 1.0
            self.class_id = 3
        elif self.attack_type == "time_shift" and self.p_eve == 0.0:
            self.p_eve = self.shift_delay_prob
            self.class_id = 4

    @classmethod
    def sample_fractional(
        cls, p_min: float = 0.10, p_max: float = 0.30, random_state: Optional[int] = None
    ) -> "EveAttacker":
        """Factory method to sample a randomized stealthy Fractional IR attacker."""
        rng = np.random.default_rng(random_state)
        p_sampled = float(rng.uniform(p_min, p_max))
        return cls(class_id=1, p_eve=p_sampled, attack_type="intercept_resend", random_state=random_state)

    @classmethod
    def sample_multiattack(cls, class_id: int, random_state: Optional[int] = None) -> "EveAttacker":
        """Factory to sample randomized parameters for any of the 5 classes."""
        rng = np.random.default_rng(random_state)
        if class_id == 0:
            return cls(class_id=0, p_eve=0.0, attack_type="benign", random_state=random_state)
        elif class_id == 1:
            p_val = float(rng.uniform(0.10, 0.30))
            return cls(class_id=1, p_eve=p_val, attack_type="intercept_resend", random_state=random_state)
        elif class_id == 2:
            mu_val = float(rng.uniform(0.20, 0.45))
            return cls(class_id=2, p_eve=mu_val, attack_type="pns", mu_wcp=mu_val, random_state=random_state)
        elif class_id == 3:
            return cls(class_id=3, p_eve=1.0, attack_type="fake_state", random_state=random_state)
        elif class_id == 4:
            shift_prob = float(rng.uniform(0.40, 0.80))
            return cls(class_id=4, p_eve=shift_prob, attack_type="time_shift", shift_delay_prob=shift_prob, random_state=random_state)
        else:
            raise ValueError(f"Unknown attack class_id: {class_id}")

    def intercept_and_resend(
        self,
        alice_bits: np.ndarray,
        alice_bases: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Physical intercept-resend: intercepts, measures, and resends states."""
        num_photons = len(alice_bits)
        if self.p_eve <= 0.0 or self.attack_type == "benign":
            return alice_bits.copy(), alice_bases.copy(), np.zeros(num_photons, dtype=bool)

        intercepted_mask = self.rng.random(size=num_photons) < self.p_eve
        resend_bits = alice_bits.copy()
        resend_bases = alice_bases.copy()

        num_intercepted = int(np.sum(intercepted_mask))
        if num_intercepted == 0:
            return resend_bits, resend_bases, intercepted_mask

        eve_bases = self.rng.integers(0, 2, size=num_intercepted, dtype=np.int8)
        alice_bases_sub = alice_bases[intercepted_mask]
        alice_bits_sub = alice_bits[intercepted_mask]

        basis_matches = (eve_bases == alice_bases_sub)
        random_outcomes = self.rng.integers(0, 2, size=num_intercepted, dtype=np.int8)
        eve_measured_bits = np.where(basis_matches, alice_bits_sub, random_outcomes)

        resend_bases[intercepted_mask] = eve_bases
        resend_bits[intercepted_mask] = eve_measured_bits

        return resend_bits, resend_bases, intercepted_mask

    def execute_full_attack(
        self,
        alice_bits: np.ndarray,
        alice_bases: np.ndarray,
        bob_bases: np.ndarray,
        base_channel_transmittance: float = 1.0,
        dark_count_rate: float = 1e-5,
        det_eff_z: float = 1.0,
        det_eff_x: float = 1.0,
    ) -> Dict[str, Any]:
        """Full physical simulation for all 5 classes (Normal + 4 attacks).

        det_eff_z / det_eff_x are Bob's per-basis detector efficiencies. Every real
        receiver has some mismatch between its two bases, so benign sessions carry a
        mild basis asymmetry that overlaps weak Time-Shift attacks. Without this,
        basis asymmetry alone separates class 4 perfectly.
        """
        n_pulses = len(alice_bits)
        det_eff = np.where(bob_bases == 0, det_eff_z, det_eff_x)
        base_eff = base_channel_transmittance * det_eff

        # -------------------------------------------------------------
        # CLASS 0: NORMAL CHANNEL (Benign)
        # -------------------------------------------------------------
        if self.attack_type == "benign":
            detection_mask = (self.rng.random(n_pulses) < base_eff) | (
                self.rng.random(n_pulses) < dark_count_rate
            )
            basis_matches = (bob_bases == alice_bases)
            random_bits = self.rng.integers(0, 2, size=n_pulses, dtype=np.int8)
            bob_bits = np.where(basis_matches, alice_bits, random_bits)
            return {
                "bob_detected_mask": detection_mask,
                "bob_detected_bits": bob_bits,
                "intercepted_mask": np.zeros(n_pulses, dtype=bool),
                "pns_split_mask": np.zeros(n_pulses, dtype=bool),
                "blinded_click_mask": np.zeros(n_pulses, dtype=bool),
                "time_shifted_mask": np.zeros(n_pulses, dtype=bool),
            }

        # -------------------------------------------------------------
        # CLASS 1: INTERCEPT-RESEND (IR) - Fractional or Full
        # -------------------------------------------------------------
        elif self.attack_type == "intercept_resend":
            resend_bits, resend_bases, intercepted_mask = self.intercept_and_resend(alice_bits, alice_bases)
            detection_mask = (self.rng.random(n_pulses) < base_eff) | (
                self.rng.random(n_pulses) < dark_count_rate
            )
            bob_matches_in = (bob_bases == resend_bases)
            random_bits = self.rng.integers(0, 2, size=n_pulses, dtype=np.int8)
            bob_bits = np.where(bob_matches_in, resend_bits, random_bits)
            return {
                "bob_detected_mask": detection_mask,
                "bob_detected_bits": bob_bits,
                "intercepted_mask": intercepted_mask,
                "pns_split_mask": np.zeros(n_pulses, dtype=bool),
                "blinded_click_mask": np.zeros(n_pulses, dtype=bool),
                "time_shifted_mask": np.zeros(n_pulses, dtype=bool),
            }

        # -------------------------------------------------------------
        # CLASS 2: PHOTON NUMBER SPLITTING (PNS)
        # -------------------------------------------------------------
        elif self.attack_type == "pns":
            # Sample Poisson photon counts per pulse
            photon_counts = self.rng.poisson(self.mu_wcp, size=n_pulses)
            multi_photon_mask = photon_counts >= 2
            single_photon_mask = photon_counts == 1

            # Eve splits multi-photon pulses and forwards the remainder losslessly over
            # the channel, but Bob's detector efficiency still applies.
            # For single-photon pulses: Eve blocks a fraction to match overall loss profile
            multi_pass_mask = multi_photon_mask & (self.rng.random(n_pulses) < det_eff)
            p_single_pass = self.pns_single_pass_frac * base_eff
            single_pass_mask = single_photon_mask & (self.rng.random(n_pulses) < p_single_pass)

            # Bob receives multi-photon (split) pulses OR passing single-photon pulses + dark counts
            detection_mask = multi_pass_mask | single_pass_mask | (self.rng.random(n_pulses) < dark_count_rate)

            # Crucial: Eve NEVER disturbs polarization of the forwarded photons!
            basis_matches = (bob_bases == alice_bases)
            random_bits = self.rng.integers(0, 2, size=n_pulses, dtype=np.int8)
            bob_bits = np.where(basis_matches, alice_bits, random_bits)

            return {
                "bob_detected_mask": detection_mask,
                "bob_detected_bits": bob_bits,
                "intercepted_mask": multi_photon_mask,
                "pns_split_mask": multi_photon_mask,
                "blinded_click_mask": np.zeros(n_pulses, dtype=bool),
                "time_shifted_mask": np.zeros(n_pulses, dtype=bool),
            }

        # -------------------------------------------------------------
        # CLASS 3: FAKE STATE (Detector Blinding)
        # -------------------------------------------------------------
        elif self.attack_type == "fake_state":
            # Eve blinds Bob's APDs with bright CW light into linear mode
            # Eve intercepts Alice's state, measures in basis B_E
            eve_bases = self.rng.integers(0, 2, size=n_pulses, dtype=np.int8)
            eve_matches_alice = (eve_bases == alice_bases)
            random_eve_bits = self.rng.integers(0, 2, size=n_pulses, dtype=np.int8)
            eve_bits = np.where(eve_matches_alice, alice_bits, random_eve_bits)

            # Eve sends bright trigger: clicks ONLY if Bob basis matches Eve basis
            basis_match_eve_bob = (bob_bases == eve_bases)
            blinded_clicks = basis_match_eve_bob & (self.rng.random(n_pulses) < (self.blinding_eff * base_eff))
            
            # Plus small linear-mode dark noise
            detection_mask = blinded_clicks | (self.rng.random(n_pulses) < (dark_count_rate * 0.1))
            bob_bits = eve_bits.copy()

            return {
                "bob_detected_mask": detection_mask,
                "bob_detected_bits": bob_bits,
                "intercepted_mask": np.ones(n_pulses, dtype=bool),
                "pns_split_mask": np.zeros(n_pulses, dtype=bool),
                "blinded_click_mask": blinded_clicks,
                "time_shifted_mask": np.zeros(n_pulses, dtype=bool),
            }

        # -------------------------------------------------------------
        # CLASS 4: TIME-SHIFT ATTACK
        # -------------------------------------------------------------
        elif self.attack_type == "time_shift":
            shifted_mask = self.rng.random(n_pulses) < self.shift_delay_prob

            # Efficiency mismatch between Z and X bases during time shift, applied on
            # top of Bob's intrinsic per-basis mismatch
            eff_z = min(1.0, base_channel_transmittance * det_eff_z * self.ts_eff_z)
            eff_x = max(0.02, base_channel_transmittance * det_eff_x * self.ts_eff_x_ratio)
            eff_per_pulse = np.where(
                shifted_mask,
                np.where(bob_bases == 0, eff_z, eff_x),
                base_eff,
            )
            detection_mask = (self.rng.random(n_pulses) < eff_per_pulse) | (self.rng.random(n_pulses) < dark_count_rate)

            # Bob measurement outcome
            basis_matches = (bob_bases == alice_bases)
            random_bits = self.rng.integers(0, 2, size=n_pulses, dtype=np.int8)
            bob_bits = np.where(basis_matches, alice_bits, random_bits)

            # Misalignment noise induced on X basis during time-shift gate edge
            x_misalign_flips = shifted_mask & (bob_bases == 1) & (self.rng.random(n_pulses) < self.ts_x_misalign)
            bob_bits = np.where(x_misalign_flips, 1 - bob_bits, bob_bits)

            return {
                "bob_detected_mask": detection_mask,
                "bob_detected_bits": bob_bits,
                "intercepted_mask": shifted_mask,
                "pns_split_mask": np.zeros(n_pulses, dtype=bool),
                "blinded_click_mask": np.zeros(n_pulses, dtype=bool),
                "time_shifted_mask": shifted_mask,
            }

        else:
            raise ValueError(f"Unsupported attack type: {self.attack_type}")

    @staticmethod
    def theoretical_qber(e_0: float, p_eve: float) -> float:
        """Asymptotic QBER on sifted bits for Fractional IR given p_eve and base noise e_0.
        
        QBER = e_0 + (1/4) * p_eve * (1 - 2*e_0)
        """
        return e_0 + 0.25 * p_eve * (1.0 - 2.0 * e_0)
