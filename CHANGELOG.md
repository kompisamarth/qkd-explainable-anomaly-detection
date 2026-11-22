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

## Section 2: Dataset generation quality

**Added**
* `src/data/dataset.py`:
  * Each session gets its own `SeedSequence` seed, so the dataset is bit-identical for any
    number of workers (tested).
  * Sessions are generated in parallel.
  * Train/test split is by session id, stratified by class.
  * CSV and Parquet export, each with a `*.meta.json` sidecar: seed, config hash, git commit,
    UTC time, class counts, parameter ranges, feature list.
  * Single-feature leakage audit for every class, flagged at AUC > 0.95.
* `scripts/06_generate_multiattack_dataset.py` uses the calibrated engine with 7 classes and
  writes three files: `train_multiattack`, `test_multiattack`, and `heldout_generalization`
  (attack intensities and 80–110 km fiber distances never used in training). It also writes
  `sessions_multiattack.csv` with per-session physics. `--smoke` produces a tiny dataset for CI;
  `engine: legacy` in `config.yaml` restores the old 5-class generator.
* `scripts/07` cross-validates with `GroupKFold` by session (audit L-2).
* `scripts/08` evaluates the test and held-out sets separately, and splits the attack-vs-normal
  comparison into its own table with the fixed 11%, per-block 11% and finite-size (Wilson)
  baselines (audit L-3).
* `DATA_CARD.md` documents every column, its unit and how it is computed.
* `tests/test_dataset_generation.py` (6 tests): worker-count invariance, seed sensitivity,
  disjoint session splits, held-out ranges, metadata sidecar, and the leakage audit flagging a
  planted leak.

**Metric changes (single seed 42; multi-seed numbers are in Section 5)**

| Metric | Old (5 classes, legacy engine) | New (7 classes, calibrated engine) | Why |
|---|---|---|---|
| Cross-validated accuracy | 0.906 (row-level StratifiedKFold) | **0.536** (GroupKFold by session) | Old folds shared overlapping windows of the same session. CV now equals the test score, which shows the leak is gone |
| Test accuracy | 0.863 | **0.536** | Two new classes, and the photon-model fingerprint is gone (below) |
| Macro-F1 | 0.856 | **0.408** | Same |
| FNR, any attack vs normal | 0.119 | **0.558** | Over 60% of attacked windows belong to PNS, Trojan horse or beam splitting, which leave Bob's QBER unchanged |
| F1 intercept-resend | 0.856 | 0.936 | QBER rise is a real physical signal |
| F1 PNS | 0.823 | **0.122** | The old PNS class was detectable through audit B-2, the missing vacuum pulses in every other class. With one Poisson source for all classes, a gain-matched PNS attack leaves no trace in single-intensity telemetry, which is the textbook result. Decoy states are needed (Section 5) |
| F1 detector blinding (old "fake state") | 0.769 | 0.570 | Blinding now only lowers QBER below e_d; the class-specific dark-count fingerprint is gone |
| F1 time-shift | 0.947 | 0.554 | The old model's artificial X-basis misalignment is gone; the real signature is a click-rate drop that benign loss steps also produce |
| F1 Trojan horse / beam splitting | n/a | 0.016 / 0.040 | Invisible to Bob's QBER telemetry by construction |
| Held-out set (unseen intensities, 80–110 km) | n/a | accuracy 0.401, FPR **0.427** | At longer distances dark counts raise benign QBER, and the model reads that as an attack: a genuine generalization failure |
| Largest single-feature AUC | 0.932 (PNS via a channel artefact) | 0.973 (intercept-resend via `multiscale_mean_w10`) | The only flag is the expected physical one (QBER for intercept-resend); every other class is ≤ 0.78 |
| Top SHAP feature | `loss_gradient` 1.29 (onset step, audit L-1) | `multiscale_mean_w10` 0.51 | Benign sessions now also contain loss steps and drift |
| Ablation, most important group | basis-dependent (−7.2 points) | channel interaction (−4.7 points); the other groups are within noise | Basis asymmetry no longer has an artificial time-shift fingerprint |

These numbers are lower than before. They are kept as they are: the old numbers measured
simulator artefacts.
