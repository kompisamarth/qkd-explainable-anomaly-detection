# Changelog

Entries follow the improvement plan, one per section. Each lists what changed, which
metrics moved, and why. "Old" numbers are the committed results at `edaa8f7`.

## Section 1: Physically calibrated simulation

**Added**
* `config/physics.yaml`: every physical parameter in one file, each with its literature
  source or an explicit `ASSUMPTION` label. `config.yaml` keeps only pipeline and ML settings;
  `src/config.py` merges the two, so older scripts read the same keys as before.
* `src/simulator/calibrated.py`: count-level weak-coherent-pulse BB84 simulator.
  * Physics: Poisson photon numbers for every class (optional signal/decoy/vacuum
    intensities); fiber loss at 0.2 dB/km or free-space log-normal fading; four detectors
    (InGaAs SPAD or SNSPD ranges) with dark counts, double clicks, optional afterpulsing and
    dead time; misalignment e_d of 1–2%.
  * Every session, attacked or not, has benign drift and loss steps.
  * Per-block Wilson 95% QBER interval.
  * Six attack models: intercept-resend, PNS, detector blinding, time-shift, Trojan horse
    (simplified) and beam splitting, documented in `docs/ATTACK_MODELS.md`.
* `src/evaluation/finite_size.py`: Wilson and Clopper–Pearson intervals.
* `tests/test_calibrated_physics.py` (11 tests):
  * no Eve and zero noise gives QBER = 0 exactly;
  * full intercept-resend gives QBER ≈ 25% within 4σ;
  * misalignment only gives QBER ≈ e_d;
  * dark counts alone give 50% error;
  * PNS is gain-matched and adds no error;
  * blinding QBER falls below the optical floor;
  * the finite-size intervals behave as expected;
  * all classes draw the channel from one distribution;
  * a seed reproduces its session.

**Fixed**
* Audit B-1 in the legacy engine: a dark-count click on a pulse that never arrived now
  carries a random bit. A block of pure dark counts went from QBER 0.000 to 0.511.

**Metric changes (legacy 3-class pipeline, scripts 01–05 and 11, rerun in full)**

| File | Change | Why |
|---|---|---|
| `classification_benchmarks_fiber.csv` | F1 down by ≤ 0.004 (largest: Hoeffding single-block −0.0039); FPR up by ≤ 0.0002 | B-1 fix: dark clicks now add errors, and drawing their random bits shifts the random-number stream for everything after |
| `cross_channel_generalization.csv` | Free-space F1 moves by ≤ 0.007; RF free-space FPR 11.6% → 12.6% | Same stream shift; the RF number was already far above the 1% target |
| `detection_latency_comparison.csv` | Mean latency moves by up to 7.5 blocks; Static 11% detection at p_eve = 0.20 drops from 40% to 27.5% | 40 sessions per point; the shift is about 1.6 binomial standard errors, i.e. sampling noise, not a physics change |
| `feature_importance_shap.csv` | magnitudes within 0.005; ranking unchanged | Same |
| `mdi_*`, `tfqkd_*`, `multiattack_*`, `trial_session_summary.csv` | identical | Only their config source moved |
