"""BB84 Quantum Key Distribution (QKD) Engine.

Supports high-performance vectorized simulation of chronological blocks and sessions,
as well as circuit-level Qiskit quantum circuit verification.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from src.simulator.noise_models import FiberNoiseModel, FreeSpaceNoiseModel
from src.simulator.eve_attacks import EveAttacker


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
    block_classes: np.ndarray        # Per-block ground truth class (0, 1, or 2)


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
    ) -> BlockResult:
        """Simulates one block of BB84 transmission with vectorized NumPy speed.
        
        Steps:
        1. Alice samples random bits in {0, 1} and random bases in {0 (Z), 1 (X)}.
        2. Eve intercepts pulses according to her attack strategy.
        3. Bob samples random measurement bases in {0, 1}.
        4. Bob measures the received states.
        5. Channel noise is applied.
        6. Basis sifting retains pulses where Alice and Bob bases coincided.
        7. Block QBER is calculated.
        """
        # Step 1: Alice state preparation
        alice_bits = self.rng.integers(0, 2, size=self.raw_photons, dtype=np.int8)
        alice_bases = self.rng.integers(0, 2, size=self.raw_photons, dtype=np.int8)

        # Step 2: Eve interception & re-emission
        resend_bits, resend_bases, intercepted_mask = eve.intercept_and_resend(
            alice_bits, alice_bases
        )

        # Step 3: Bob basis selection
        bob_bases = self.rng.integers(0, 2, size=self.raw_photons, dtype=np.int8)

        # Step 4: Bob physical measurement
        # If Bob's basis matches incoming photon basis, he detects resend_bits deterministically.
        # If Bob's basis differs from incoming photon basis, outcome is uniformly 0 or 1.
        bob_matches_in = (bob_bases == resend_bases)
        random_outcomes = self.rng.integers(0, 2, size=self.raw_photons, dtype=np.int8)
        bob_detected_bits = np.where(bob_matches_in, resend_bits, random_outcomes)

        # Step 5: Channel noise
        effective_noise = channel_model.get_block_qber(block_idx)
        bob_final_bits = channel_model.apply_noise(bob_detected_bits, effective_noise)

        # Step 6: Basis Sifting (Alice basis == Bob basis)
        sift_mask = (alice_bases == bob_bases)
        sifted_alice = alice_bits[sift_mask]
        sifted_bob = bob_final_bits[sift_mask]
        num_sifted = len(sifted_alice)

        # Step 7: QBER calculation
        if num_sifted > 0:
            errors = int(np.sum(sifted_alice != sifted_bob))
            block_qber = float(errors / num_sifted)
        else:
            errors = 0
            block_qber = 0.0

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
    ) -> SessionSequence:
        """Simulates an entire chronological session with a dynamic regime shift at attack_start_block.
        
        Blocks 1 .. (attack_start_block - 1) are strictly Benign (Class 0).
        At attack_start_block, Eve activates with target_attack_class:
            - Class 0: Remains Benign throughout all B blocks
            - Class 1: Full IR (p_eve = 1.0)
            - Class 2: Fractional IR (p_eve in [0.10, 0.30])
        """
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

        # Determine Eve's attack parameters
        if target_attack_class == 0:
            attack_p_eve = 0.0
            eve_attack = EveAttacker(class_id=0, p_eve=0.0, random_state=int(self.rng.integers(1e9)))
        elif target_attack_class == 1:
            attack_p_eve = 1.0
            eve_attack = EveAttacker(class_id=1, p_eve=1.0, random_state=int(self.rng.integers(1e9)))
        elif target_attack_class == 2:
            if p_eve_fractional is not None:
                attack_p_eve = float(p_eve_fractional)
            else:
                attack_p_eve = float(self.rng.uniform(0.10, 0.30))
            eve_attack = EveAttacker(class_id=2, p_eve=attack_p_eve, random_state=int(self.rng.integers(1e9)))
        else:
            raise ValueError(f"Invalid attack_class: {target_attack_class}")

        benign_eve = EveAttacker(class_id=0, p_eve=0.0, random_state=int(self.rng.integers(1e9)))

        qbers = np.zeros(blocks_per_session, dtype=np.float64)
        sifted_lens = np.zeros(blocks_per_session, dtype=np.int32)
        channel_noises = np.zeros(blocks_per_session, dtype=np.float64)
        block_classes = np.zeros(blocks_per_session, dtype=np.int32)

        for b in range(1, blocks_per_session + 1):
            idx = b - 1
            if b < attack_start_block or target_attack_class == 0:
                # Pre-attack benign regime
                active_eve = benign_eve
                block_classes[idx] = 0
            else:
                # Active attack regime
                active_eve = eve_attack
                block_classes[idx] = target_attack_class

            res = self.simulate_block_vectorized(b, channel, active_eve)
            qbers[idx] = res.qber
            sifted_lens[idx] = res.sifted_bits
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
        )

    def verify_with_qiskit(
        self,
        num_photons: int = 500,
        p_eve: float = 1.0,
        noise_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """Runs circuit-level Qiskit simulation for rigorous quantum physics verification.
        
        Prepares 1-qubit BB84 quantum circuits for each photon:
        - Alice encodes bit in Z or X basis (using I, X, H gates).
        - Eve intercepts, measures in her basis (collapsing the state), and
          re-prepares a fresh qubit encoding her outcome in her basis.
        - Bob measures in his chosen basis.
        - Sifts and computes empirical error rate.

        Eve's measure-and-resend is executed as two circuits (prepare+measure,
        then re-prepare+measure) so the wavefunction collapse is physically real
        rather than an algebraic cancellation of stacked H gates.
        """
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        def prepare_and_measure(bit: int, prep_basis: int, meas_basis: int) -> int:
            """Prepare |bit> in prep_basis, measure in meas_basis, return one-shot outcome."""
            qc = QuantumCircuit(1)
            if bit == 1:
                qc.x(0)
            if prep_basis == 1:
                qc.h(0)
            # Rotating the X basis back onto Z before a computational-basis readout
            if meas_basis == 1:
                qc.h(0)
            probs = Statevector.from_instruction(qc).probabilities()
            return int(self.rng.choice([0, 1], p=probs / probs.sum()))

        alice_bits = self.rng.integers(0, 2, size=num_photons)
        alice_bases = self.rng.integers(0, 2, size=num_photons)  # 0: Z, 1: X
        bob_bases = self.rng.integers(0, 2, size=num_photons)

        sifted_alice = []
        sifted_bob = []

        for i in range(num_photons):
            bit_in, basis_in = int(alice_bits[i]), int(alice_bases[i])

            # Eve attack (with probability p_eve): measure, collapse, re-prepare
            if self.rng.random() < p_eve:
                eve_basis = int(self.rng.integers(0, 2))
                bit_in = prepare_and_measure(bit_in, basis_in, eve_basis)
                basis_in = eve_basis

            measured_bit = prepare_and_measure(bit_in, basis_in, int(bob_bases[i]))

            # Noise bit flip
            if noise_rate > 0.0 and self.rng.random() < noise_rate:
                measured_bit = 1 - measured_bit

            # Sifting
            if alice_bases[i] == bob_bases[i]:
                sifted_alice.append(alice_bits[i])
                sifted_bob.append(measured_bit)

        sifted_alice = np.array(sifted_alice)
        sifted_bob = np.array(sifted_bob)
        empirical_qber = float(np.mean(sifted_alice != sifted_bob)) if len(sifted_alice) > 0 else 0.0

        return {
            "num_photons": num_photons,
            "sifted_count": len(sifted_alice),
            "empirical_qber": empirical_qber,
            "theoretical_qber": EveAttacker.theoretical_qber(noise_rate, p_eve),
        }
