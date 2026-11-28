"""One-command reproduction of every result in results/.

    python reproduce.py            # full pipeline (~2.5 h on a 24-core desktop; the SVM dominates)
    python reproduce.py --quick    # smoke run (~3 min): tests + tiny dataset + 2-seed evaluation
    python reproduce.py --legacy   # also rerun the original 3-class pipeline (scripts 02-05)

Steps run in order and stop at the first failure. The physics unit tests run first, so
a broken simulator (QBER != 0 without noise, != 25% under full intercept-resend, != e_d
with misalignment only) fails the pipeline before any dataset is generated.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def step(name, args):
    print(f"\n{'=' * 78}\n>> {name}\n   {' '.join(args)}\n{'=' * 78}", flush=True)
    t0 = time.time()
    r = subprocess.run([PY, "-W", "ignore", *args], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"FAILED: {name} (exit {r.returncode})")
    print(f"   done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--phase3", action="store_true", help="isolated Phase 3 reproduction")
    a = ap.parse_args()

    step("Unit tests (physics, datasets, features, dashboard)", ["-m", "pytest", "-q"])
    if a.phase3:
        step("Phase 3 immutable run", ["scripts/13_source_side_pipeline.py"] + (["--quick"] if a.quick else []))
        return
    step("Legacy physics verification", ["scripts/01_verify_physics.py"])
    if a.quick:
        step("Smoke dataset", ["scripts/06_generate_multiattack_dataset.py", "--smoke",
                               "--out-dir", str(ROOT / "data" / "processed" / "smoke")])
        step("Smoke multi-seed evaluation", ["scripts/10_multiseed_evaluation.py", "--seeds", "1", "2",
                                             "--sessions-per-class", "6", "--heldout-sessions-per-class", "3",
                                             "--svm-max-rows", "1500", "--tag", "smoke"])
        print("\nQuick run finished. Its metrics are NOT the published ones; run without --quick to regenerate them.")
        return
    if a.legacy:
        for s in ("02_generate_dataset", "03_run_benchmarks", "04_explain_and_plot", "05_modern_qkd_tfqkd"):
            step(f"Legacy {s}", [f"scripts/{s}.py"])
    step("Calibrated dataset (train / test / held-out)", ["scripts/06_generate_multiattack_dataset.py"])
    step("Train XGBoost (GroupKFold CV)", ["scripts/07_train_multiattack_classifiers.py"])
    step("Evaluate checkpoint (test + held-out)", ["scripts/08_evaluate_multiattack.py"])
    step("SHAP and feature-group ablation", ["scripts/09_explain_and_ablation.py"])
    step("Multi-seed RF / XGBoost / SVM vs thresholds", ["scripts/10_multiseed_evaluation.py",
                                                         "--seeds", "42", "43", "44", "45", "46",
                                                         "--sessions-per-class", "200"])
    step("Multi-seed with decoy states", ["scripts/10_multiseed_evaluation.py", "--seeds", "42", "43", "44", "45",
                                          "46", "--sessions-per-class", "200", "--decoy"])
    step("MDI-QKD comparison and trial sessions", ["scripts/11_mdi_protocol_comparison.py"])
    step("Real-data calibration and OOD test", ["scripts/12_real_data_evaluation.py"])
    print("\nAll steps finished. Start the dashboard with:  python run_dashboard.py")


if __name__ == "__main__":
    main()
