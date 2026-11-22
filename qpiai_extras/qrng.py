"""Random bits for Alice from the QpiAI QRNG, with a NumPy fallback.

`QRNGSource` exposes the one method the pulse-level simulator needs from a NumPy
Generator, `integers(0, 2, size, dtype)`, so it can replace Alice's bit and basis
source:

    from qpiai_extras import QRNGSource
    sim = BB84Simulator(raw_photons_per_block=1024, random_state=1)
    sim.alice_source = QRNGSource(fallback_seed=1)      # bits + bases from the QRNG
    seq = sim.simulate_session(1, 1, attack_mode="multiattack")

Randomness caveat (from the SDK's own documentation): on the local simulator
"QpiAI-QSV-Local" the QRNG samples with NumPy, so it is NOT genuinely quantum random.
Only a QPU device gives hardware quantum randomness. `source_label` records which
one produced the bits.
"""

import logging
from typing import Optional

import numpy as np

from qpiai_extras.backend import qpiai_status

log = logging.getLogger("qpiai_extras")


class QRNGSource:
    def __init__(self, device_name: str = "QpiAI-QSV-Local", n_bits: int = 16,
                 fallback_seed: Optional[int] = None, prefetch: int = 65536):
        self.device_name = device_name
        self.n_bits = int(n_bits)
        self.prefetch = int(prefetch)
        self._fallback = np.random.default_rng(fallback_seed)
        self._buffer = np.zeros(0, dtype=np.int8)
        self.status = qpiai_status(login=True)
        self.source_label = (f"QpiAI QRNG on {device_name}" if self.status.usable
                             else "NumPy PCG64 fallback (QpiAI unavailable)")
        if device_name.endswith("Local") and self.status.usable:
            log.info("QRNG on %s samples with NumPy inside the SDK: pseudo-random, not quantum.", device_name)

    def _fetch_qpiai(self, n: int) -> np.ndarray:
        from qpiai_quantum.algorithms import QRNG
        shots = int(np.ceil(n / self.n_bits))
        values = QRNG(n_bits=self.n_bits).generate(shots=max(shots, 2), output_format="bitstring",
                                                    device_name=self.device_name)
        bits = np.frombuffer("".join(values).encode(), dtype=np.uint8) - ord("0")
        # generate() returns samples grouped by outcome; shuffle so order carries no structure
        self._fallback.shuffle(bits)
        return bits.astype(np.int8)

    def bits(self, n: int) -> np.ndarray:
        """n random bits in {0, 1}."""
        if not self.status.usable:
            return self._fallback.integers(0, 2, size=n, dtype=np.int8)
        while len(self._buffer) < n:
            try:
                self._buffer = np.concatenate([self._buffer, self._fetch_qpiai(max(n, self.prefetch))])
            except Exception as exc:
                log.warning("QpiAI QRNG call failed (%s); switching to NumPy fallback.", type(exc).__name__)
                self.status.usable = False
                self.source_label = "NumPy PCG64 fallback (QpiAI call failed)"
                return self._fallback.integers(0, 2, size=n, dtype=np.int8)
        out, self._buffer = self._buffer[:n], self._buffer[n:]
        return out

    # NumPy Generator compatibility for the simulator hook
    def integers(self, low, high=None, size=None, dtype=np.int64):
        if (low, high) not in ((0, 2), (2, None)):
            raise ValueError("QRNGSource only produces bits: integers(0, 2, size)")
        n = int(np.prod(size)) if size is not None else 1
        b = self.bits(n).astype(dtype)
        return b.reshape(size) if size is not None else b[0]
