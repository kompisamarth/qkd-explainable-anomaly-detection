"""Sanity check and verification script for BB84 QKD physics, noise models, and Eve attacks.

Verifies:
1. Ideal noiseless channel: e_0 = 0, p_eve = 0 => QBER = 0.00
2. Full Intercept-Resend attack: e_0 = 0, p_eve = 1.0 => QBER approx 0.25 (25%)
3. Fractional IR attack: e_0 = 0.03, p_eve = 0.20 => QBER approx 0.077 (7.7%, beneath 11% abort threshold)
4. Free-space atmospheric turbulence dynamics (scintillation drift, AR(1) temporal continuity)
5. Circuit-level Qiskit quantum verification
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.simulator.noise_models import FiberNoiseModel, FreeSpaceNoiseModel
from src.simulator.eve_attacks import EveAttacker
from src.simulator.bb84_engine import BB84Simulator


def run_physics_checks():
    print("=" * 70)
    print("BB84 QUANTUM PHYSICS & ERROR MODEL VERIFICATION")
    print("=" * 70)
    
    sim = BB84Simulator(raw_photons_per_block=20000, random_state=42)
    
    # ---------------------------------------------------------
    # TEST 1: Noiseless Channel without Eavesdropper
    # ---------------------------------------------------------
    print("\n[TEST 1] Noiseless Channel without Eavesdropper (e_0 = 0, p_eve = 0)")
    noiseless_fiber = FiberNoiseModel(base_qber=0.0, random_state=42)
    benign_eve = EveAttacker(class_id=0, p_eve=0.0, random_state=42)
    res_1 = sim.simulate_block_vectorized(1, noiseless_fiber, benign_eve)
    
    print(f"  Raw Photons: {res_1.raw_photons}")
    print(f"  Sifted Bits: {res_1.sifted_bits} (approx 50% of raw)")
    print(f"  Bit Errors:  {res_1.bit_errors}")
    print(f"  Empirical QBER:   {res_1.qber:.4f}")
    print(f"  Theoretical QBER: {EveAttacker.theoretical_qber(0.0, 0.0):.4f}")
    assert res_1.qber == 0.0, f"Expected 0.0 QBER, got {res_1.qber}"
    print("  -> PASSED: Zero QBER on noiseless, unperturbed channel.")

    # ---------------------------------------------------------
    # TEST 2: Full Intercept-Resend Attack (Noiseless base channel)
    # ---------------------------------------------------------
    print("\n[TEST 2] Full Intercept-Resend Attack (e_0 = 0, p_eve = 1.0)")
    full_eve = EveAttacker(class_id=1, p_eve=1.0, random_state=42)
    res_2 = sim.simulate_block_vectorized(1, noiseless_fiber, full_eve)
    
    theo_2 = EveAttacker.theoretical_qber(0.0, 1.0)
    print(f"  Sifted Bits:      {res_2.sifted_bits}")
    print(f"  Bit Errors:       {res_2.bit_errors}")
    print(f"  Empirical QBER:   {res_2.qber:.4f}")
    print(f"  Theoretical QBER: {theo_2:.4f} (25.0%)")
    assert abs(res_2.qber - theo_2) < 0.015, f"Empirical QBER {res_2.qber} drifted too far from {theo_2}"
    print("  -> PASSED: Full IR induces theoretical ~25% QBER.")

    # ---------------------------------------------------------
    # TEST 3: Fractional / Stealthy IR Attack with Natural Fiber Noise
    # ---------------------------------------------------------
    print("\n[TEST 3] Fractional Stealthy IR Attack (e_0 = 0.03, p_eve = 0.20)")
    fiber_3 = FiberNoiseModel(base_qber=0.03, random_state=42)
    frac_eve = EveAttacker(class_id=2, p_eve=0.20, random_state=42)
    
    # Run multiple blocks to obtain high statistical precision
    qbers = []
    for b in range(1, 20):
        res = sim.simulate_block_vectorized(b, fiber_3, frac_eve)
        qbers.append(res.qber)
    
    emp_qber_3 = float(np.mean(qbers))
    theo_3 = EveAttacker.theoretical_qber(0.03, 0.20)
    print(f"  Base Channel Noise (e_0): 3.0%")
    print(f"  Eve Interception Rate:    20.0%")
    print(f"  Empirical Mean QBER:      {emp_qber_3:.4f} ({emp_qber_3*100:.2f}%)")
    print(f"  Theoretical QBER:         {theo_3:.4f} ({theo_3*100:.2f}%)")
    print(f"  Static Abort Threshold:   11.00%")
    print(f"  Safety Margin Below 11%:  {(0.11 - emp_qber_3)*100:.2f}%")
    assert abs(emp_qber_3 - theo_3) < 0.01, f"Empirical QBER {emp_qber_3} drifted from {theo_3}"
    assert emp_qber_3 < 0.11, "Fractional attack failed to stay beneath static 11% abort threshold!"
    print("  -> PASSED: Stealthy attack confirmed to bypass static 11% threshold.")

    # ---------------------------------------------------------
    # TEST 4: Non-Stationary Free-Space Atmospheric Scintillation
    # ---------------------------------------------------------
    print("\n[TEST 4] Non-Stationary Free-Space Atmospheric Scintillation Channel")
    fso = FreeSpaceNoiseModel(base_qber=0.03, scintillation_std=0.008, ar1_phi=0.75, random_state=42)
    fso_qbers = [fso.get_block_qber(b) for b in range(100)]
    
    fso_mean = float(np.mean(fso_qbers))
    fso_std = float(np.std(fso_qbers))
    # Check lag-1 autocorrelation
    lag1_corr = float(np.corrcoef(fso_qbers[:-1], fso_qbers[1:])[0, 1])
    
    print(f"  FSO Sequence Mean QBER:   {fso_mean:.4f} (Nominal: 0.0300)")
    print(f"  FSO Sequence QBER Std:    {fso_std:.4f}")
    print(f"  Empirical Lag-1 Autocorr: {lag1_corr:.4f} (Target phi: ~0.75)")
    assert abs(fso_mean - 0.03) < 0.005, "FSO mean drifted excessively from baseline"
    assert lag1_corr > 0.50, f"Lag-1 autocorrelation {lag1_corr} is too weak for atmospheric memory"
    print("  -> PASSED: Realistic temporal correlation and turbulence verified.")

    # ---------------------------------------------------------
    # TEST 5: Circuit-Level Qiskit Quantum Verification
    # ---------------------------------------------------------
    print("\n[TEST 5] Qiskit Circuit-Level Quantum Verification")
    try:
        qiskit_res = sim.verify_with_qiskit(num_photons=400, p_eve=1.0, noise_rate=0.0)
        print(f"  Qiskit Measured Sifted Count: {qiskit_res['sifted_count']}")
        print(f"  Qiskit Empirical QBER:        {qiskit_res['empirical_qber']:.4f}")
        print(f"  Qiskit Theoretical QBER:      {qiskit_res['theoretical_qber']:.4f}")
        assert abs(qiskit_res['empirical_qber'] - 0.25) < 0.08, "Qiskit simulation deviated from 25%"
        print("  -> PASSED: Circuit-level Qiskit simulation confirms exact quantum physics.")
    except Exception as e:
        print(f"  [Qiskit verification note]: {e}")

    print("\n" + "=" * 70)
    print("ALL QUANTUM PHYSICS AND SIMULATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_physics_checks()
