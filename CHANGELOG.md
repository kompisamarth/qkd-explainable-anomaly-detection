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
| Largest single-feature AUC | 0.932 (mean of the per-seed maxima; last seed: time-shift via `basis_variance_ratio`, 0.920) | 0.973 (intercept-resend via `multiscale_mean_w10`) | The only flag is the expected physical one (QBER for intercept-resend); every other class is ≤ 0.78 |
| Top SHAP feature | `loss_gradient` 1.29 (onset step, audit L-1) | `multiscale_mean_w10` 0.51 | Benign sessions now also contain loss steps and drift |
| Ablation, most important group | basis-dependent (−7.2 points) | channel interaction (−4.7 points); the other groups are within noise | Basis asymmetry no longer has an artificial time-shift fingerprint |

These numbers are lower than before. They are kept as they are: the old numbers measured
simulator artefacts.

## Section 3: Open-source and real data

**Added**
* `DATASETS.md` surveys Zenodo, IEEE DataPort and Kaggle (Figshare's API could not be queried;
  stated). It covers 5 real-telemetry datasets and 6 synthetic ones, with links and licences.
  **No public dataset contains labelled attacks on real hardware.**
* Your untracked `data/raw/` files were identified as Zenodo 21792844 (Horgan et al., TCD /
  CONNECT testbed, CC-BY-4.0). They are now cited; `--download` fetches the remaining files.
* `src/data/real_telemetry.py` maps the real CSVs onto the order-independent part of the feature
  schema, because the real rows are sorted by QBER and carry no timestamps.
* `scripts/12_real_data_evaluation.py` (outputs labelled "Real data (benign, Zenodo 21792844)"):
  * calibration of real against simulated benign QBER versus loss;
  * an out-of-distribution false-alarm test.
* `tests/test_real_telemetry.py` (3 tests).

**Results (real data, reported separately)**
* Calibration: the real Toshiba MU system without noise injection runs at 3.3–7.7% QBER over
  10–30 dB. The simulator gives 2.2–6.0% (InGaAs SPAD) and 1.6% (SNSPD) at the same losses. The
  real system's intrinsic error is about 3%, above the 1–2% literature range in `physics.yaml`.
  This is reported, not tuned away.
* OOD test: the simulation-trained model flags **all 8** local real benign blocks as
  intercept-resend (false-alarm rate 100%); the fixed 11% threshold flags none. The model does
  not transfer to this hardware without recalibrating its benign baseline.

## Section 4: QpiAI Quantum SDK integration (optional, isolated)

**Added** `qpiai_extras/`. Nothing in the main pipeline imports it.
* `backend.py` looks for an API key in `QPIAI_API_KEY`, then `API_KEY`, then a `qcloud.env`
  file. It logs only where the key came from, never the key itself. If the SDK is not installed,
  there is no key, or login fails, it falls back with a clear log message.
* `qrng.py`: `QRNGSource` provides Alice's bits from the QpiAI QRNG, or NumPy as a fallback. It
  plugs into the pulse-level simulator through the new `BB84Simulator.alice_source` hook. As the
  SDK's own documentation states, the local `QpiAI-QSV-Local` QRNG is pseudo-random; only a QPU
  device gives quantum randomness.
* `bb84_circuit.py`: circuit-level BB84 with Z/X preparation and an intercept-resend Eve whose
  measurement is a separate circuit run. It is validated against the NumPy simulator:

  | | Circuit QBER | Theory | NumPy simulator |
  |---|---|---|---|
  | No Eve | 0.000 | 0.000 | 0.000 |
  | Full intercept-resend (193 sifted rounds) | 0.264 | 0.250 ± 0.031 | 0.250 |

  These were run on the fallback backend, because no QpiAI key was available here.
* `figures.py`: Bloch-sphere and intercept-resend circuit figures (matplotlib and Qiskit fallback).
* `run_demo.py` and `qpiai_bb84_demo.ipynb`; `requirements.txt` pins `qpiai-quantum==0.8.0` as an
  optional extra.
* `.gitignore` now ignores `qcloud.env` and `*.env`.
* `tests/test_qpiai_extras.py` (4 tests): key lookup order, the key never appears in the status,
  the NumPy fallback, and the circuit fallback physics.

No existing metric changed.

## Section 6: Dashboard

**Backend (`backend/server.py`)**
* Runs on the calibrated engine with 7 classes, with physics descriptions rewritten to match the
  new models.
* New `POST /api/simulate/custom`: protocol (BB84 / decoy BB84), channel, fiber distance,
  detector, attack, intensity, onset and seed. It flags settings outside the training range.
* New `GET /api/channel` (fiber vs free-space) and `GET /api/data` (dataset sidecars, class
  balance, distance histograms, leakage audits, real-data calibration and OOD results).
* `GET /api/benchmarks` now serves the multi-seed comparison, per-class metrics, confusion
  matrix, latency, reliability, held-out and decoy results.
* Every result carries a `provenance` block (label, source files, seeds) and lists missing files.
* The SHAP tab's threshold panel gains a finite-size (Wilson lower bound) check.

**Frontend**
* Eight tabs: Overview, Simulation (Blind / Guided / Custom), Classifier & SHAP, Benchmarks,
  Channels, MDI-QKD, Data, Threat Atlas.
* Simulation: per-block QBER with a shaded 95% Wilson band, window mean, and the 11% line.
* Benchmarks:
  * FNR/FPR bars with 95% CI whiskers for RF, XGBoost, SVM-RBF and the three threshold baselines;
  * per-class F1 with ±1 sd whiskers and a confusion-matrix heatmap;
  * decoy vs no-decoy F1, latency vs intensity, reliability curves, held-out results, ablation.
* Channels: fiber vs free-space FPR and F1 with CIs.
* Data: dataset metadata, class balance, distance histograms per split, leakage table, parameter
  ranges, and a separately labelled real-data panel.
* Every chart shows a provenance caption: **SIMULATED DATA**, **REAL DATA** or **ANALYTIC
  MODEL**, plus source file and seed.
* The stale hard-coded claims (85.6% accuracy, 99.4% vs 11.5% FNR, "five regimes") are removed;
  headline numbers are read from the multi-seed summary.
* `scripts/11` trial sessions now use the calibrated engine (7 classes).

**Metric changes:** `trial_session_summary.csv` now has 7 rows from the calibrated engine. The
MDI-QKD outputs are unchanged.

## Section 5: ML evaluation

**Added**
* `scripts/10_multiseed_evaluation.py`, rewritten. For each of 5 seeds (42–46, 200 sessions per
  class) it generates a fresh session-split dataset, a held-out set and a free-space set, then
  trains Random Forest, XGBoost and SVM-RBF. The SVM trains on a session-stratified subsample of
  at most 12,000 rows.
* Every metric is reported as mean, sd and a 95% Student-t interval, clipped to [0, 1] for
  rates: accuracy, macro-F1, per-class precision/recall/F1, confusion matrix (summed), binary
  FNR (primary) and FPR, binary AUROC/AUPRC, macro one-vs-rest AUROC, top-label ECE, binary
  Brier score, reliability curves, latency vs intensity tertile, pre-onset false-alarm rate,
  held-out and free-space results, and a per-seed leakage audit.
* Three threshold baselines: fixed 11% on the window mean, fixed 11% on one block, and
  finite-size 11% (Wilson lower bound above 11%).
* `--decoy` reruns everything with signal/decoy/vacuum intensities and the decoy features
  (`*_decoy` files).
* `src/evaluation/metrics.py`: binary rates, CI, ECE, reliability, latency.

**Results (simulated, 5 seeds, mean [95% CI])**

| | Old (5 classes, legacy engine) | New, 7 classes | New, 7 classes + decoy |
|---|---|---|---|
| XGBoost accuracy | 0.856 ± 0.014 (sd) | **0.514** [0.497, 0.531] | — |
| XGBoost macro-F1 | 0.847 | **0.393** [0.372, 0.413] | **0.613** |
| RF / SVM macro-F1 | not evaluated | 0.375 / 0.330 | 0.609 / 0.458 |
| XGBoost FNR (missed attacks) | 0.115 | **0.573** [0.551, 0.596] | 0.350 |
| XGBoost FPR (false alarms) | 0.077 | 0.155 [0.139, 0.170] | 0.105 |
| Fixed 11% FNR / FPR | 0.994 / 0 | **1.000 / 0** (all three variants) | 1.000 / 0 |
| Binary AUROC (XGBoost) | not reported | 0.661 | — |
| Top-label ECE (XGBoost) | not reported | 0.106 | — |
| Largest single-feature AUC | 0.932 | 0.976 (intercept-resend via QBER, the physical signal) | — |

Per-class F1 for XGBoost, without and with decoy states:

| Class | No decoy | Decoy |
|---|---|---|
| Normal | 0.59 | 0.71 |
| Intercept-resend | 0.93 | 0.95 |
| PNS | **0.08** | **0.77** |
| Blinding | 0.58 | 0.79 |
| Time-shift | 0.47 | 0.54 |
| Trojan horse | 0.03 | 0.53 |
| Beam splitting | 0.05 | 0.02 |

* **Why the numbers fell.** Most of the drop from the old result comes from removing artefacts:
  the photon-model fingerprint, stationary benign channels, and row-level CV. The rest comes from
  adding two attacks that are invisible by design. The new numbers are kept as they are.
* **Decoy states.** They make PNS detectable (0.08 → 0.77 F1), exactly as decoy-state theory
  predicts.
* **Trojan horse.** Its jump to 0.53 F1 under decoys is checked in a paired-seed comparison.
  With SNSPDs, Eve's leaked light makes vacuum pulses click where honest dark counts almost never
  do; with InGaAs SPADs the excess is small (median ×1.10). This depends entirely on the
  *assumed* leak ratio (`forward_leak_ratio`, 1e-5–1e-4) and on there being no other background,
  such as Raman noise. Treat it as an upper bound.
* **Beam splitting** stays undetectable, as it should.
* **Latency (XGBoost).**
  * Intercept-resend: detected in 91% of low-intensity sessions (mean 8 blocks) and 100% of
    mid/high ones (≈ 0 blocks).
  * Blinding: 70–100% detected, latency 38 → 11 blocks as intensity rises.
  * Time-shift: 90–97% detected, 15 → 7 blocks.
  * PNS, Trojan horse and beam splitting: 12–49% "detected". With a 15% pre-onset false-alarm
    rate per block, that is mostly chance.
* **Held-out set** (unseen intensities, 80–110 km): XGBoost accuracy 0.404, FNR 0.458,
  **FPR 0.364**. The model does not extrapolate to longer distances.
* **Channel generalization** (fiber-trained, free-space tested): XGBoost FPR rises from 0.155 to
  **0.503**, RF from 0.095 to 0.523. Log-normal fading looks like an attack to every model.
  The 11% thresholds raise no alarms on either channel, so they never false-alarm but also
  never catch anything.
* **SHAP and ablation** (`scripts/09`) were regenerated on the session-grouped split in Section 2.

## Section 7: Repository and reproducibility

**Added**
* `reproduce.py` runs the whole pipeline with one command, tests first, stopping at the first
  failure. `--quick` is a ~1.5 min smoke run whose outputs carry a `_smoke` suffix and are
  git-ignored, so published results are never overwritten. `--legacy` also reruns scripts 02–05.
  A `Makefile` wraps these as `all`, `quick`, `legacy`, `test`, `install` and `dashboard`.
* `requirements-lock.txt`: the exact versions behind `results/` (Python 3.13.7).
  `requirements.txt` keeps loose bounds and now lists `pytest` and `httpx`.
* `LICENSE` (MIT, matching the README badge), `CITATION.cff` (including the Zenodo dataset
  reference), and `LIMITATIONS.md`.
* `.github/workflows/ci.yml` runs on every push, on Python 3.11 and 3.13: the unit tests, then a
  smoke dataset, then a smoke evaluation. The physics checks fail the build.
* `scripts/10 --tag` adds an output suffix.
* README:
  * a new Section 0 with current results, document links and reproduction steps;
  * a refreshed architecture and quickstart;
  * the old Phase 2 numbers marked as superseded;
  * limitations pointing to `LIMITATIONS.md`.
* `.gitignore` covers the generated Parquet datasets, their sidecars, smoke outputs and `qcloud.env`.

**Verified locally:** `python reproduce.py --quick` passes (59 tests, smoke dataset, 2-seed
evaluation) without touching committed results. The GitHub Actions workflow has not run yet;
it will run on the first push.

No metric changed.
