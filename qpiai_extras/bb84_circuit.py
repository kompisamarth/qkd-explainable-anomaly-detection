"""Circuit-level BB84 (Z/X preparation, measurement, intercept-resend Eve).

Runs a few hundred rounds as real circuits and compares the sifted QBER with the
NumPy simulator's analytic value e_0 + p_eve/4 (1 - 2 e_0). Rounds are batched: each
circuit carries up to `qubits_per_circuit` independent rounds (one qubit each), run for
one shot. Eve's measure-and-resend is two circuit executions (her measurement, then her
re-preparation and Bob's measurement), so the collapse is a real measurement.

Backends, in order of preference:
  "qpiai"   QpiAI SDK Circuit.run on `device_name` (needs an API key, see backend.py)
  "exact"   fallback: Born-rule sampling of the same single-qubit circuits with NumPy
"""

import logging
from typing import Dict, Optional

import numpy as np

from qpiai_extras.backend import qpiai_status

log = logging.getLogger("qpiai_extras")


def _exact_measure(rng, bits, prep_basis, meas_basis):
    """|bit> in prep basis, measured in meas basis: deterministic if bases match, else 50/50."""
    same = prep_basis == meas_basis
    return np.where(same, bits, rng.integers(0, 2, size=len(bits)))


def _qpiai_measure(bits, prep_basis, meas_basis, device_name, qubits_per_circuit):
    from qpiai_quantum import Circuit
    out = np.zeros(len(bits), dtype=int)
    for start in range(0, len(bits), qubits_per_circuit):
        idx = range(start, min(start + qubits_per_circuit, len(bits)))
        n = len(idx)
        c = Circuit(n, n)
        for q, i in enumerate(idx):
            if bits[i]:
                c.x(q)
            if prep_basis[i]:
                c.h(q)
            if meas_basis[i]:
                c.h(q)          # rotate X basis onto Z before readout
            c.measure(q, q)
        counts = c.run(shots=1, experiment_name="bb84-validation", device_name=device_name).get_counts()
        bitstring = next(iter(counts))
        # SDK bitstrings are big-endian over classical bits (clbit 0 leftmost) unless reverse_bits
        vals = [int(ch) for ch in bitstring[-n:]]
        out[start:start + n] = vals
    return out


def run_bb84_rounds(rounds: int = 400, p_eve: float = 1.0, noise_rate: float = 0.0, seed: int = 0,
                    device_name: str = "QpiAI-QSV-Local", qubits_per_circuit: int = 16,
                    force_fallback: bool = False) -> Dict:
    rng = np.random.default_rng(seed)
    status = None if force_fallback else qpiai_status(login=True)
    use_qpiai = bool(status and status.usable)
    backend = f"qpiai:{device_name}" if use_qpiai else "exact (NumPy Born-rule fallback)"

    def measure(bits, pb, mb):
        if use_qpiai:
            return _qpiai_measure(bits, pb, mb, device_name, qubits_per_circuit)
        return _exact_measure(rng, bits, pb, mb)

    a_bits = rng.integers(0, 2, rounds)
    a_bases = rng.integers(0, 2, rounds)
    b_bases = rng.integers(0, 2, rounds)
    eve = rng.random(rounds) < p_eve
    in_bits, in_bases = a_bits.copy(), a_bases.copy()
    if eve.any():
        e_bases = rng.integers(0, 2, int(eve.sum()))
        in_bits[eve] = measure(a_bits[eve], a_bases[eve], e_bases)   # Eve measures: real collapse
        in_bases[eve] = e_bases                                        # and re-prepares in her basis
    bob = measure(in_bits, in_bases, b_bases)
    flips = rng.random(rounds) < noise_rate
    bob = np.where(flips, 1 - bob, bob)

    sift = a_bases == b_bases
    n = int(sift.sum())
    qber = float(np.mean(a_bits[sift] != bob[sift])) if n else 0.0
    theory = noise_rate + 0.25 * p_eve * (1 - 2 * noise_rate)
    se = np.sqrt(theory * (1 - theory) / max(n, 1))
    return {"backend": backend, "rounds": rounds, "sifted": n, "p_eve": p_eve, "qber": qber,
            "theory": theory, "std_error": float(se), "within_3_sigma": bool(abs(qber - theory) <= 3 * se + 1e-12),
            "qpiai_reason": None if status is None else status.reason}


def compare_with_numpy_simulator(rounds: int = 400, seed: int = 0, **kw) -> Dict:
    """Circuit QBER vs the vectorised NumPy simulator for no Eve and full intercept-resend."""
    from src.simulator.bb84_engine import BB84Simulator
    from src.simulator.eve_attacks import EveAttacker
    from src.simulator.noise_models import FiberNoiseModel
    rows = {}
    for p in (0.0, 1.0):
        circ = run_bb84_rounds(rounds, p_eve=p, seed=seed, **kw)
        sim = BB84Simulator(raw_photons_per_block=200_000, random_state=seed)
        eve = EveAttacker(class_id=1 if p else 0, p_eve=p, attack_type="intercept_resend" if p else "benign",
                          random_state=seed + 1)
        blk = sim.simulate_block_vectorized(1, FiberNoiseModel(0.0, random_state=seed + 2), eve)
        rows[f"p_eve={p}"] = {"circuit": circ, "numpy_simulator_qber": blk.qber}
    return rows
