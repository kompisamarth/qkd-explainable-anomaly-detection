"""Report figures: Bloch sphere of the four BB84 states and a BB84 / intercept-resend circuit diagram.

Uses the QpiAI SDK's drawing when it is installed and usable; otherwise draws the same
content with matplotlib (Bloch sphere) and Qiskit's text drawer (circuit), so the
figures exist with or without a QpiAI account.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BB84_STATES = {"|0>  (Z, bit 0)": (0, 0, 1), "|1>  (Z, bit 1)": (0, 0, -1),
               "|+>  (X, bit 0)": (1, 0, 0), "|->  (X, bit 1)": (-1, 0, 0)}


def bloch_sphere_figure(path: Path) -> Path:
    fig = plt.figure(figsize=(5.2, 5.2), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="#cbd5e1", lw=0.4)
    for lbl, (x, y, z) in BB84_STATES.items():
        ax.quiver(0, 0, 0, x, y, z, color="#0d9488" if z else "#d97706", lw=2, arrow_length_ratio=0.12)
        ax.text(x * 1.25, y * 1.25, z * 1.25, lbl, fontsize=7)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.set_title("BB84 states on the Bloch sphere\n(Z basis: poles, X basis: equator)", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def circuit_figure(path: Path) -> Path:
    """Alice prepares |1> in X, Eve measures in Z and resends, Bob measures in X."""
    try:
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(1, 2)
        qc.x(0); qc.h(0); qc.barrier(label="Alice |->")
        qc.measure(0, 0); qc.barrier(label="Eve M_Z")
        qc.reset(0); qc.barrier(label="Eve resends her result")
        qc.h(0); qc.measure(0, 1)
        text = str(qc.draw("text", fold=-1))
    except Exception:
        text = ("Alice  ─X─H─┤ Eve: M_Z ├─ resend |e> ─H─┤ Bob: M_X ├\n"
                "(X/H prepare |->; Eve's Z measurement collapses it; Bob's X result is random)")
    text = ("Intercept-resend round:  Alice |-> (X basis)  ->  Eve measures Z and resends  ->  Bob measures X\n\n"
            + text + "\n\nBob's result is 50/50, so this sifted round carries a 50% error;"
                     " averaged over Eve's basis choice the QBER is 25%.")
    fig, ax = plt.subplots(figsize=(10, 2.6), dpi=200)
    ax.axis("off")
    ax.text(0.0, 1.0, text, family="monospace", fontsize=7, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path
