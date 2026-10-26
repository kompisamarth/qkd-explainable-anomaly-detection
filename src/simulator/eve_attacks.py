"""Eavesdropping attack models for discrete-variable BB84 QKD.

Implements Benign (Class 0), Full Intercept-Resend (Class 1), and
Fractional/Stealthy Intercept-Resend (Class 2) attacks.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class EveAttackConfig:
    """Configuration parameters for an active eavesdropper."""
    class_id: int            # 0: Benign, 1: Full IR, 2: Fractional IR
    p_eve: float             # Interception probability per pulse
    name: str                # Human-readable label


class EveAttacker:
    """Models an eavesdropper executing an Intercept-Resend (IR) attack on BB84 pulses.
    
    In an Intercept-Resend attack:
    1. Eve intercepts a transmitted photon with probability p_eve.
    2. Eve measures the photon in a randomly chosen basis B_E in {0 (Z), 1 (X)}.
    3. Eve prepares a new single photon in the state corresponding to her measurement outcome
       and forwards it to Bob.
    4. Photons not intercepted (probability 1 - p_eve) transit unaltered.
    """

    def __init__(self, class_id: int = 0, p_eve: float = 0.0, random_state: Optional[int] = None):
        self.class_id = int(class_id)
        self.p_eve = float(p_eve)
        self.rng = np.random.default_rng(random_state)
        
        if self.class_id == 0:
            self.name = "Benign"
            self.p_eve = 0.0
        elif self.class_id == 1:
            self.name = "Full Intercept-Resend"
            self.p_eve = 1.0
        elif self.class_id == 2:
            self.name = "Fractional Intercept-Resend"
        else:
            self.name = f"Custom IR (p={self.p_eve:.2f})"

    @classmethod
    def sample_fractional(cls, p_min: float = 0.10, p_max: float = 0.30, random_state: Optional[int] = None) -> "EveAttacker":
        """Factory method to sample a randomized stealthy Fractional IR attacker."""
        rng = np.random.default_rng(random_state)
        p_sampled = float(rng.uniform(p_min, p_max))
        return cls(class_id=2, p_eve=p_sampled, random_state=random_state)

    def intercept_and_resend(
        self,
        alice_bits: np.ndarray,
        alice_bases: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulates Eve's physical interception, basis measurement, and re-emission.
        
        Args:
            alice_bits: Array of binary values (0 or 1) prepared by Alice.
            alice_bases: Array of basis choices (0 for Z-basis, 1 for X-basis).
            
        Returns:
            resend_bits: The bit values re-emitted toward Bob.
            resend_bases: The bases in which the re-emitted states are prepared.
            intercepted_mask: Boolean array indicating which pulses were intercepted.
        """
        num_photons = len(alice_bits)
        if self.p_eve <= 0.0:
            # Benign channel: all photons pass untouched
            intercepted_mask = np.zeros(num_photons, dtype=bool)
            return alice_bits.copy(), alice_bases.copy(), intercepted_mask

        # Eve samples which pulses to intercept
        intercepted_mask = self.rng.random(size=num_photons) < self.p_eve
        
        # Start with unmodified transmission
        resend_bits = alice_bits.copy()
        resend_bases = alice_bases.copy()
        
        num_intercepted = int(np.sum(intercepted_mask))
        if num_intercepted == 0:
            return resend_bits, resend_bases, intercepted_mask
        
        # Eve randomly selects measurement basis {0 (Z), 1 (X)} for intercepted pulses
        eve_bases = self.rng.integers(0, 2, size=num_intercepted)
        alice_bases_sub = alice_bases[intercepted_mask]
        alice_bits_sub = alice_bits[intercepted_mask]
        
        # Determine Eve's measurement outcomes:
        # If Eve's basis matches Alice's, measurement is deterministic (exact match).
        # If Eve's basis differs from Alice's, outcome is uniformly 0 or 1 (50% random flip).
        basis_matches = (eve_bases == alice_bases_sub)
        random_outcomes = self.rng.integers(0, 2, size=num_intercepted)
        eve_measured_bits = np.where(basis_matches, alice_bits_sub, random_outcomes)
        
        # Eve re-prepares photon in her measured state (eve_bases, eve_measured_bits)
        resend_bases[intercepted_mask] = eve_bases
        resend_bits[intercepted_mask] = eve_measured_bits
        
        return resend_bits, resend_bases, intercepted_mask

    @staticmethod
    def theoretical_qber(e_0: float, p_eve: float) -> float:
        """Computes the theoretical asymptotic QBER on sifted bits for a given p_eve and base noise e_0.
        
        For each sifted bit (where Alice and Bob used the same basis):
        - With prob (1 - p_eve), photon was not intercepted -> error with prob e_0
        - With prob p_eve, photon was intercepted ->
            - Eve matched basis with prob 1/2 -> no extra error -> error with prob e_0
            - Eve mismatched basis with prob 1/2 -> Bob gets random bit (prob 1/2 error)
            - Total error given interception: 1/4 + (1/2)*e_0
        Combining: QBER = e_0 + (1/4) * p_eve * (1 - 2*e_0)
        """
        return e_0 + 0.25 * p_eve * (1.0 - 2.0 * e_0)
