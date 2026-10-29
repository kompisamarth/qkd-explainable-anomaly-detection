"""BB84 Quantum Key Distribution (QKD) Engine.

Supports high-performance vectorized simulation of chronological blocks and sessions,
as well as circuit-level Qiskit quantum circuit verification.
Supports 5 major operational classes (Normal + 4 Major Attack Archetypes).
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from src.simulator.noise_models import FiberNoiseModel, FreeSpaceNoiseModel
from src.simulator.eve_attacks import EveAttacker


# ----------------------------------------------------------------------------
# Per-session nuisance ranges (README section 3.0, task P2.1).
#
# These ranges are deliberately OVERLAPPING across classes. Before this change
# every class ran at a fixed channel transmittance of 1.0 and a fixed dark-count
# rate, so `transmission_efficiency` collapsed onto a per-class constant
# (1.00 benign/IR, ~0.26 PNS, ~0.48 fake-state, ~0.70 time-shift) and separated
# three of the five classes on its own. The classifier was then learning the
# simulator's fingerprints rather than attack physics.
#
# Each range is chosen so that a benign session can look like a weak attack:
#   - benign base QBER reaches down to 0.5%, overlapping fake-state blinding residual
#   - benign channel loss spans 10-95%, overlapping PNS yield suppression
#   - benign detectors carry up to 30% basis-efficiency mismatch, overlapping
#     weak time-shift
#   - IR interception reaches down to 5%, where the induced error is comparable
#     to benign baseline drift
# ----------------------------------------------------------------------------
NUISANCE_RANGES = {
    "base_qber": (0.005, 0.060),            # e_0: short well-aligned link .. noisy long link
    "channel_transmittance": (0.10, 0.95),  # distance / coupling loss
    "log10_dark_count": (-6.0, -4.3),       # 1e-6 .. ~5e-5 per pulse
    "det_eff_z": (0.80, 1.00),              # Bob's Z-basis detector efficiency
    "det_eff_x_ratio": (0.70, 1.00),        # intrinsic (benign) basis mismatch
    "ir_p_eve": (0.05, 0.30),               # fractional interception
    "pns_mu": (0.20, 0.45),                 # WCP mean photon number
    "pns_single_pass": (0.20, 0.60),        # fraction of single-photon pulses Eve forwards
    "blinding_eff": (0.70, 0.98),           # how reliably Eve's faked states trigger clicks
    "blinding_residual_qber": (0.002, 0.020),  # Eve's state-preparation imperfection
    "ts_shift_prob": (0.30, 0.85),          # fraction of pulses time-shifted
    "ts_eff_x_ratio": (0.30, 0.90),         # attack-induced basis mismatch
    "ts_x_misalign": (0.04, 0.16),          # gate-edge misalignment error
}


@dataclass
class BlockResult:
    """Summary metrics of a single transmitted BB84 block."""
    block_index: int
    raw_photons: int
    sifted_bits: int
    bit_errors: int
    qber: float
    effective_channel_noise: float
    p_eve: float
    class_id: int
    intercepted_count: int
    # Rich metrics for the 63 physics-informed features
    sifted_z: int = 0
    errors_z: int = 0
    qber_z: float = 0.0
    sifted_x: int = 0
    errors_x: int = 0
    qber_x: float = 0.0
    detection_count: int = 0
    transmission_efficiency: float = 0.50
    error_gap_mean: float = 0.0
    error_gap_std: float = 0.0
    error_gap_cv: float = 0.0
    error_clustering: float = 0.0


@dataclass
class SessionSequence:
    """Complete multi-block chronological transmission session."""
    session_id: int
    channel_type: str
    attack_class: int
    p_eve: float
    attack_start_block: int
    block_qbers: np.ndarray          # Array of block QBER values of length B
    block_sifted_lens: np.ndarray    # Array of sifted bit lengths per block
    block_channel_noises: np.ndarray # Instantaneous channel noise per block
    block_classes: np.ndarray        # Per-block ground truth class
    # Rich Phase 2 arrays
    block_qbers_z: Optional[np.ndarray] = None
    block_qbers_x: Optional[np.ndarray] = None
    block_sifted_z: Optional[np.ndarray] = None
    block_sifted_x: Optional[np.ndarray] = None
    block_raw_photons: Optional[np.ndarray] = None
    block_detection_counts: Optional[np.ndarray] = None
    block_transmission_effs: Optional[np.ndarray] = None
    block_error_gap_means: Optional[np.ndarray] = None
    block_error_gap_stds: Optional[np.ndarray] = None
    block_error_gap_cvs: Optional[np.ndarray] = None
    block_error_clusterings: Optional[np.ndarray] = None


class BB84Simulator:
    """Core BB84 discrete block transmission simulator."""

    def __init__(
        self,
        raw_photons_per_block: int = 1024,
        base_qber: float = 0.03,
        random_state: Optional[int] = None,
    ):
        self.raw_photons = int(raw_photons_per_block)
        self.base_qber = float(base_qber)
        self.rng = np.random.default_rng(random_state)

    def simulate_block_vectorized(
        self,
        block_idx: int,
        channel_model: Union[FiberNoiseModel, FreeSpaceNoiseModel],
        eve: EveAttacker,
        channel_transmittance: float = 1.0,
        dark_count_rate: float = 1e-5,
        det_eff_z: float = 1.0,
        det_eff_x: float = 1.0,
    ) -> BlockResult:
        """Simulates one block of BB84 transmission with vectorized NumPy speed.

        The channel/detector arguments default to the original lossless values so
        direct callers and tests are unchanged; `simulate_session` supplies
        randomized per-session values when `randomize_nuisance=True`.
        """
        # Step 1: Alice samples bits and bases
        alice_bits = self.rng.integers(0, 2, size=self.raw_photons, dtype=np.int8)
        alice_bases = self.rng.integers(0, 2, size=self.raw_photons, dtype=np.int8)

        # Step 2: Bob basis selection
        bob_bases = self.rng.integers(0, 2, size=self.raw_photons, dtype=np.int8)

        # Step 3: Eve attack and channel propagation
        effective_noise = channel_model.get_block_qber(block_idx)

        # Use full physical attack model for 5-class simulation
        attack_res = eve.execute_full_attack(
            alice_bits=alice_bits,
            alice_bases=alice_bases,
            bob_bases=bob_bases,
            base_channel_transmittance=channel_transmittance,
            dark_count_rate=dark_count_rate,
            det_eff_z=det_eff_z,
            det_eff_x=det_eff_x,
        )

        detection_mask = attack_res["bob_detected_mask"]
        bob_detected_bits = attack_res["bob_detected_bits"]
        intercepted_mask = attack_res["intercepted_mask"]

        # Step 4: Channel background depolarizing noise applied to detected pulses
        if eve.attack_type == "fake_state":
            # Bob's detector is in classical linear mode, so the channel's
            # single-photon depolarizing noise no longer applies. The residual error
            # is set by how well Eve prepares and controls her faked states.
            effective_noise = eve.blinding_residual_qber
        bob_final_bits = channel_model.apply_noise(bob_detected_bits, effective_noise)

        # Step 5: Sifting (Alice basis == Bob basis AND photon was detected)
        sift_mask = (alice_bases == bob_bases) & detection_mask
        sifted_alice = alice_bits[sift_mask]
        sifted_bob = bob_final_bits[sift_mask]
        num_sifted = len(sifted_alice)

        # Step 6: Overall and basis-dependent QBER calculation
        if num_sifted > 0:
            sifted_errors = (sifted_alice != sifted_bob).astype(np.int8)
            errors = int(np.sum(sifted_errors))
            block_qber = float(errors / num_sifted)

            # Error-gap statistics within the block
            err_pos = np.where(sifted_errors == 1)[0]
            if len(err_pos) >= 2:
                gaps = np.diff(err_pos)
                gap_mean = float(np.mean(gaps))
                gap_std = float(np.std(gaps))
                gap_cv = float(gap_std / (gap_mean + 1e-9))
                consecutive_pairs = int(np.sum(gaps == 1))
                clustering = float(consecutive_pairs / max(len(err_pos), 1))
            else:
                gap_mean = float(num_sifted) if len(err_pos) == 0 else float(num_sifted / 2)
                gap_std = 0.0
                gap_cv = 0.0
                clustering = 0.0
        else:
            errors = 0
            block_qber = 0.0
            gap_mean = 0.0
            gap_std = 0.0
            gap_cv = 0.0
            clustering = 0.0

        # Basis breakdown
        sift_z = sift_mask & (alice_bases == 0)
        sift_x = sift_mask & (alice_bases == 1)
        num_z = int(np.sum(sift_z))
        num_x = int(np.sum(sift_x))

        if num_z > 0:
            err_z = int(np.sum(alice_bits[sift_z] != bob_final_bits[sift_z]))
            qber_z = float(err_z / num_z)
        else:
            err_z = 0
            qber_z = 0.0

        if num_x > 0:
            err_x = int(np.sum(alice_bits[sift_x] != bob_final_bits[sift_x]))
            qber_x = float(err_x / num_x)
        else:
            err_x = 0
            qber_x = 0.0

        num_detected = int(np.sum(detection_mask))
        trans_eff = float(num_detected / max(self.raw_photons, 1))

        return BlockResult(
            block_index=block_idx,
            raw_photons=self.raw_photons,
            sifted_bits=num_sifted,
            bit_errors=errors,
            qber=block_qber,
            effective_channel_noise=effective_noise,
            p_eve=eve.p_eve,
            class_id=eve.class_id,
            intercepted_count=int(np.sum(intercepted_mask)),
            sifted_z=num_z,
            errors_z=err_z,
            qber_z=qber_z,
            sifted_x=num_x,
            errors_x=err_x,
            qber_x=qber_x,
            detection_count=num_detected,
            transmission_efficiency=trans_eff,
            error_gap_mean=gap_mean,
            error_gap_std=gap_std,
            error_gap_cv=gap_cv,
            error_clustering=clustering,
        )

    def simulate_session(
        self,
        session_id: int,
        target_attack_class: int,
        channel_type: str = "fiber",
        blocks_per_session: int = 200,
        attack_start_block: int = 101,
        p_eve_fractional: Optional[float] = None,
        fiber_base_qber: float = 0.03,
        scintillation_std: float = 0.30,
        ar1_phi: float = 0.75,
        attack_mode: str = "auto",  # 'auto', 'legacy', or 'multiattack'
        randomize_nuisance: bool = False,
    ) -> SessionSequence:
        """Simulates an entire chronological session with a dynamic regime shift.
        
        Supports both legacy mode (0: Benign, 1: Full IR, 2: Fractional IR) and
        Phase 2 multi-attack mode (0: Normal, 1: Fractional IR, 2: PNS, 3: Fake State, 4: Time-Shift).
        """
        # Auto-detect mode if not explicitly set
        if attack_mode == "auto":
            if target_attack_class in [3, 4]:
                attack_mode = "multiattack"
            else:
                attack_mode = "legacy"

        # ---- Per-session nuisance parameters (P2.1) ----
        # Drawn from the deliberately overlapping NUISANCE_RANGES so that no class
        # is separable by a single deterministic feature. When disabled, the original
        # lossless defaults are used and behaviour is unchanged.
        nz = NUISANCE_RANGES
        if randomize_nuisance:
            u = self.rng.uniform
            fiber_base_qber = float(u(*nz["base_qber"]))
            channel_transmittance = float(u(*nz["channel_transmittance"]))
            dark_count_rate = float(10.0 ** u(*nz["log10_dark_count"]))
            det_eff_z = float(u(*nz["det_eff_z"]))
            det_eff_x = float(det_eff_z * u(*nz["det_eff_x_ratio"]))
            atk_kwargs = {
                "pns_single_pass_frac": float(u(*nz["pns_single_pass"])),
                "blinding_eff": float(u(*nz["blinding_eff"])),
                "blinding_residual_qber": float(u(*nz["blinding_residual_qber"])),
                "ts_eff_x_ratio": float(u(*nz["ts_eff_x_ratio"])),
                "ts_x_misalign": float(u(*nz["ts_x_misalign"])),
            }
            ir_range = nz["ir_p_eve"]
            pns_range = nz["pns_mu"]
            ts_range = nz["ts_shift_prob"]
        else:
            channel_transmittance, dark_count_rate = 1.0, 1e-5
            det_eff_z = det_eff_x = 1.0
            atk_kwargs = {}
            ir_range, pns_range, ts_range = (0.10, 0.30), (0.20, 0.45), (0.50, 0.85)

        # Instantiate channel noise model
        if channel_type == "fiber":
            channel = FiberNoiseModel(base_qber=fiber_base_qber, random_state=int(self.rng.integers(1e9)))
        elif channel_type == "free_space":
            channel = FreeSpaceNoiseModel(
                base_qber=fiber_base_qber,
                scintillation_std=scintillation_std,
                ar1_phi=ar1_phi,
                random_state=int(self.rng.integers(1e9)),
            )
        else:
            raise ValueError(f"Unknown channel_type: {channel_type}")

        # Configure Eve
        seed_eve = int(self.rng.integers(1e9))
        if attack_mode == "legacy":
            if target_attack_class == 0:
                attack_p_eve = 0.0
                eve_attack = EveAttacker(class_id=0, p_eve=0.0, random_state=seed_eve)
            elif target_attack_class == 1:
                attack_p_eve = 1.0
                eve_attack = EveAttacker(class_id=1, p_eve=1.0, random_state=seed_eve)
            elif target_attack_class == 2:
                attack_p_eve = float(p_eve_fractional) if p_eve_fractional is not None else float(self.rng.uniform(0.10, 0.30))
                eve_attack = EveAttacker(class_id=1, p_eve=attack_p_eve, random_state=seed_eve)
            else:
                raise ValueError(f"Invalid legacy attack_class: {target_attack_class}")
        else:  # multiattack mode (5 classes)
            if target_attack_class == 0:
                attack_p_eve = 0.0
                eve_attack = EveAttacker(class_id=0, p_eve=0.0, random_state=seed_eve)
            elif target_attack_class == 1:  # Fractional IR
                attack_p_eve = float(p_eve_fractional) if p_eve_fractional is not None else float(self.rng.uniform(*ir_range))
                eve_attack = EveAttacker(class_id=1, p_eve=attack_p_eve, random_state=seed_eve, **atk_kwargs)
            elif target_attack_class == 2:  # PNS
                attack_p_eve = float(self.rng.uniform(*pns_range))
                eve_attack = EveAttacker(class_id=2, p_eve=attack_p_eve, mu_wcp=attack_p_eve, random_state=seed_eve, **atk_kwargs)
            elif target_attack_class == 3:  # Fake State
                attack_p_eve = 1.0
                eve_attack = EveAttacker(class_id=3, p_eve=1.0, random_state=seed_eve, **atk_kwargs)
            elif target_attack_class == 4:  # Time-Shift
                attack_p_eve = float(self.rng.uniform(*ts_range))
                eve_attack = EveAttacker(class_id=4, p_eve=attack_p_eve, shift_delay_prob=attack_p_eve, random_state=seed_eve, **atk_kwargs)
            else:
                raise ValueError(f"Invalid multiattack class: {target_attack_class}")

        benign_eve = EveAttacker(class_id=0, p_eve=0.0, random_state=int(self.rng.integers(1e9)))

        # Preallocate block arrays
        qbers = np.zeros(blocks_per_session, dtype=np.float64)
        qbers_z = np.zeros(blocks_per_session, dtype=np.float64)
        qbers_x = np.zeros(blocks_per_session, dtype=np.float64)
        sifted_lens = np.zeros(blocks_per_session, dtype=np.int32)
        sifted_z = np.zeros(blocks_per_session, dtype=np.int32)
        sifted_x = np.zeros(blocks_per_session, dtype=np.int32)
        raw_photons = np.full(blocks_per_session, self.raw_photons, dtype=np.int32)
        det_counts = np.zeros(blocks_per_session, dtype=np.int32)
        trans_effs = np.zeros(blocks_per_session, dtype=np.float64)
        gap_means = np.zeros(blocks_per_session, dtype=np.float64)
        gap_stds = np.zeros(blocks_per_session, dtype=np.float64)
        gap_cvs = np.zeros(blocks_per_session, dtype=np.float64)
        clusterings = np.zeros(blocks_per_session, dtype=np.float64)
        channel_noises = np.zeros(blocks_per_session, dtype=np.float64)
        block_classes = np.zeros(blocks_per_session, dtype=np.int32)

        for b in range(1, blocks_per_session + 1):
            idx = b - 1
            if b < attack_start_block or target_attack_class == 0:
                active_eve = benign_eve
                block_classes[idx] = 0
            else:
                active_eve = eve_attack
                block_classes[idx] = target_attack_class

            res = self.simulate_block_vectorized(
                b, channel, active_eve,
                channel_transmittance=channel_transmittance,
                dark_count_rate=dark_count_rate,
                det_eff_z=det_eff_z,
                det_eff_x=det_eff_x,
            )
            qbers[idx] = res.qber
            qbers_z[idx] = res.qber_z
            qbers_x[idx] = res.qber_x
            sifted_lens[idx] = res.sifted_bits
            sifted_z[idx] = res.sifted_z
            sifted_x[idx] = res.sifted_x
            det_counts[idx] = res.detection_count
            trans_effs[idx] = res.transmission_efficiency
            gap_means[idx] = res.error_gap_mean
            gap_stds[idx] = res.error_gap_std
            gap_cvs[idx] = res.error_gap_cv
            clusterings[idx] = res.error_clustering
            channel_noises[idx] = res.effective_channel_noise

        return SessionSequence(
            session_id=session_id,
            channel_type=channel_type,
            attack_class=target_attack_class,
            p_eve=attack_p_eve,
            attack_start_block=attack_start_block,
            block_qbers=qbers,
            block_sifted_lens=sifted_lens,
            block_channel_noises=channel_noises,
            block_classes=block_classes,
            block_qbers_z=qbers_z,
            block_qbers_x=qbers_x,
            block_sifted_z=sifted_z,
            block_sifted_x=sifted_x,
            block_raw_photons=raw_photons,
            block_detection_counts=det_counts,
            block_transmission_effs=trans_effs,
            block_error_gap_means=gap_means,
            block_error_gap_stds=gap_stds,
            block_error_gap_cvs=gap_cvs,
            block_error_clusterings=clusterings,
        )

    def verify_with_qiskit(
        self,
        num_photons: int = 500,
        p_eve: float = 1.0,
        noise_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """Validates BB84 intercept-resend physics using gate-level Qiskit circuits."""
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        def prepare_and_measure(bit: int, prep_basis: int, meas_basis: int) -> int:
            qc = QuantumCircuit(1)
            if bit == 1:
                qc.x(0)
            if prep_basis == 1:
                qc.h(0)
            if meas_basis == 1:
                qc.h(0)
            probs = Statevector.from_instruction(qc).probabilities()
            return int(self.rng.choice([0, 1], p=probs / probs.sum()))

        alice_bits = self.rng.integers(0, 2, size=num_photons)
        alice_bases = self.rng.integers(0, 2, size=num_photons)
        bob_bases = self.rng.integers(0, 2, size=num_photons)

        sifted_alice = []
        sifted_bob = []

        for i in range(num_photons):
            bit_in, basis_in = int(alice_bits[i]), int(alice_bases[i])

            if self.rng.random() < p_eve:
                eve_basis = int(self.rng.integers(0, 2))
                bit_in = prepare_and_measure(bit_in, basis_in, eve_basis)
                basis_in = eve_basis

            measured_bit = prepare_and_measure(bit_in, basis_in, int(bob_bases[i]))

            if noise_rate > 0.0 and self.rng.random() < noise_rate:
                measured_bit = 1 - measured_bit

            if alice_bases[i] == bob_bases[i]:
                sifted_alice.append(alice_bits[i])
                sifted_bob.append(measured_bit)

        sifted_alice = np.array(sifted_alice)
        sifted_bob = np.array(sifted_bob)
        num_sifted = len(sifted_alice)
        qber = float(np.mean(sifted_alice != sifted_bob)) if num_sifted > 0 else 0.0

        return {
            "num_photons": num_photons,
            "sifted_count": num_sifted,
            "empirical_qber": qber,
            "theoretical_qber": EveAttacker.theoretical_qber(noise_rate, p_eve),
        }
