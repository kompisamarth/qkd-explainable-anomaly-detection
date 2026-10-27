"""Noise models for BB84 Quantum Channels: Fiber (stationary) and Free-Space (scintillation).

Physically models depolarizing background error in optical fiber vs.
non-stationary atmospheric turbulence with temporally correlated log-normal drift.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ChannelNoiseState:
    """Represents the instantaneous noise parameters of the quantum channel."""
    nominal_qber: float
    effective_qber: float
    channel_type: str


class FiberNoiseModel:
    """Stationary depolarizing channel typical of telecom optical fiber.
    
    In standard optical fiber (1550 nm), background error is primarily governed by
    detector dark counts, polarization drift compensation residuals, and optical
    cross-talk. This produces a stationary background QBER e_0 (typically 2-4%).
    """

    def __init__(self, base_qber: float = 0.03, random_state: Optional[int] = None):
        self.base_qber = float(base_qber)
        self.rng = np.random.default_rng(random_state)
        self.channel_type = "fiber"

    def get_block_qber(self, block_idx: int) -> float:
        """Returns the stationary baseline QBER for the current transmission block."""
        return self.base_qber

    def apply_noise(self, bits: np.ndarray, block_qber: Optional[float] = None) -> np.ndarray:
        """Applies independent bit-flip channel noise according to block_qber."""
        p_err = self.base_qber if block_qber is None else block_qber
        flips = self.rng.random(size=len(bits)) < p_err
        return np.bitwise_xor(bits, flips.astype(int))


class FreeSpaceNoiseModel:
    """Non-stationary atmospheric channel subject to turbulence and scintillation.
    
    For satellite-to-ground or terrestrial free-space optical (FSO) links, atmospheric
    turbulence causes beam wandering, beam spreading, and intensity scintillation.
    This creates temporally correlated fluctuations in received photon intensity and
    signal-to-noise ratio, modeled here via an AR(1) log-normal drift process.
    """

    def __init__(
        self,
        base_qber: float = 0.03,
        scintillation_std: float = 0.30,
        ar1_phi: float = 0.75,
        min_qber: float = 0.005,
        max_qber: float = 0.15,
        random_state: Optional[int] = None,
    ):
        self.base_qber = float(base_qber)
        self.scintillation_std = float(scintillation_std)
        self.ar1_phi = float(ar1_phi)
        self.min_qber = float(min_qber)
        self.max_qber = float(max_qber)
        self.rng = np.random.default_rng(random_state)
        self.channel_type = "free_space"
        
        # State of the AR(1) log-intensity disturbance
        self._current_ar_state = 0.0

    def reset(self) -> None:
        """Resets the temporal disturbance process state."""
        self._current_ar_state = 0.0

    def get_block_qber(self, block_idx: int) -> float:
        """Evolves the AR(1) atmospheric turbulence and returns current effective QBER.
        
        z_t = phi * z_{t-1} + sqrt(1 - phi^2) * eps_t, where eps_t ~ N(0, sigma^2)
        e_t = e_0 * exp(z_t - 0.5 * sigma^2)
        """
        innovation = self.rng.normal(0.0, self.scintillation_std)
        self._current_ar_state = (
            self.ar1_phi * self._current_ar_state
            + np.sqrt(1.0 - self.ar1_phi**2) * innovation
        )
        
        # Log-normal scaling around baseline
        qber_drift = self.base_qber * np.exp(
            self._current_ar_state - 0.5 * (self.scintillation_std**2)
        )
        
        # Clamp within realistic physical boundaries
        return float(np.clip(qber_drift, self.min_qber, self.max_qber))

    def apply_noise(self, bits: np.ndarray, block_qber: float) -> np.ndarray:
        """Applies channel noise governed by the instantaneous block QBER."""
        flips = self.rng.random(size=len(bits)) < block_qber
        return np.bitwise_xor(bits, flips.astype(int))
