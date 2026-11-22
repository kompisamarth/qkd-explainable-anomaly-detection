"""Run the optional QpiAI demo end to end (works without an account via fallbacks).

    python -m qpiai_extras.run_demo

Writes results/metrics/qpiai_bb84_validation.json and results/figures/fig_bloch_bb84.png,
fig_bb84_ir_circuit.png. The JSON records which backend actually ran.
"""

import json
import logging
from pathlib import Path

import numpy as np

from qpiai_extras.backend import qpiai_status
from qpiai_extras.bb84_circuit import compare_with_numpy_simulator
from qpiai_extras.figures import bloch_sphere_figure, circuit_figure
from qpiai_extras.qrng import QRNGSource

ROOT = Path(__file__).resolve().parent.parent


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    st = qpiai_status(login=False)
    print(f"QpiAI SDK installed: {st.sdk_installed}; API key found: {st.api_key_found} ({st.key_source})")

    q = QRNGSource(fallback_seed=0)
    bits = q.bits(20000)
    print(f"QRNG source: {q.source_label}; mean of 20,000 bits = {bits.mean():.4f}")

    res = compare_with_numpy_simulator(rounds=400, seed=0)
    for k, v in res.items():
        c = v["circuit"]
        print(f"{k}: circuit QBER {c['qber']:.4f} (theory {c['theory']:.4f}, ±{c['std_error']:.4f}, "
              f"{c['sifted']} sifted, {c['backend']}) | NumPy simulator {v['numpy_simulator_qber']:.4f}")

    from src.simulator.bb84_engine import BB84Simulator
    sim = BB84Simulator(raw_photons_per_block=4096, random_state=3)
    sim.alice_source = q
    seq = sim.simulate_session(1, 1, attack_mode="multiattack", blocks_per_session=60, attack_start_block=31,
                               p_eve_fractional=0.25)
    print(f"Simulator with QRNG-driven Alice: QBER pre {seq.block_qbers[:30].mean():.4f}, "
          f"post {seq.block_qbers[30:].mean():.4f}")

    figs = ROOT / "results" / "figures"
    bloch_sphere_figure(figs / "fig_bloch_bb84.png")
    circuit_figure(figs / "fig_bb84_ir_circuit.png")
    out = {"qpiai_status": st.__dict__, "qrng_source": q.source_label, "qrng_bit_mean": float(bits.mean()),
           "validation": res,
           "simulator_with_qrng": {"qber_pre": float(seq.block_qbers[:30].mean()),
                                   "qber_post": float(seq.block_qbers[30:].mean())}}
    (ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "results" / "metrics" / "qpiai_bb84_validation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("Wrote results/metrics/qpiai_bb84_validation.json and two figures.")


if __name__ == "__main__":
    main()
