# Explainable ML for Eavesdropping Detection and Device-Assumption Monitoring in QKD

[![tests](https://github.com/kompisamarth/qkd-explainable-anomaly-detection/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE Format](https://img.shields.io/badge/Paper-IEEE%20EL%20Conference-orange.svg)](#)

A research pipeline and benchmark implementation for explainable, low-latency machine learning detection of stealthy attacks on Quantum Key Distribution. It covers the traditional **BB84** protocol, a **Sending-or-Not-Sending Twin-Field QKD (SNS-TF-QKD)** simulator, and — from Phase 3 — a **source-side side-channel monitor** targeting the transmitter light-injection events that can violate the assumptions of a particular source model.

**Project status:** Phase 1 and Phase 2 complete. The 2026-11 revision replaced the multi-attack
simulator with a physically calibrated one and re-evaluated everything; see
[Section 0](#0-current-results-and-one-command-reproduction). Phase 3 now includes a local source-monitor prototype. See [the implementation and reproduction guide](docs/PHASE3.md). The historical plan is in
[Section 3](#3-phase-3-roadmap).

> **All results are from simulation** except the files prefixed `real_`, which use benign
> telemetry from commercial QKD systems (Zenodo 21792844). Read [LIMITATIONS.md](LIMITATIONS.md)
> before quoting any number.

---

## 0. Current Results and One-Command Reproduction

| Document | What it covers |
|---|---|
| [AUDIT.md](AUDIT.md) | Bugs, leakage risks and unrealistic parameters found before the revision, and where each CSV comes from |
| [CHANGELOG.md](CHANGELOG.md) | Every change, section by section, with the metrics it moved and why |
| [config/physics.yaml](config/physics.yaml) | Every physical parameter, with its literature source or an explicit ASSUMPTION label |
| [docs/ATTACK_MODELS.md](docs/ATTACK_MODELS.md) | The six attack models and what each one simplifies |
| [DATA_CARD.md](DATA_CARD.md) | Every dataset column, its unit and how it is computed |
| [DATASETS.md](DATASETS.md) | Survey of public QKD datasets; how the real Zenodo telemetry is used |
| [LIMITATIONS.md](LIMITATIONS.md) | Simulated data, simplified attacks, asymptotic assumptions, no hardware validation |

### Reproduce

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-lock.txt   # exact versions used for results/ (Python 3.13)
python reproduce.py                    # everything, ~2.5 h on 24 cores (or: make all)
python reproduce.py --quick            # tests + smoke dataset + 2-seed evaluation, ~3 min
python run_dashboard.py                # http://127.0.0.1:8000
```

`reproduce.py` runs the unit tests first. If the physics checks fail (QBER ≠ 0 without noise,
≠ 25% under full intercept-resend, ≠ e_d with misalignment only), it stops before generating
data. GitHub Actions runs the tests, a smoke dataset and a smoke evaluation on every push.

### Headline results (simulated, 7 classes, 5 seeds, mean and 95% CI)

| Detector (attack vs normal) | Missed attacks (FNR) | False alarms (FPR) | 7-class macro-F1 |
|---|---|---|---|
| Fixed 11% QBER threshold (window, block, or finite-size Wilson) | 100% | 0% | — |
| XGBoost | 57.3% [55.1, 59.6] | 15.5% [13.9, 17.0] | 0.393 [0.372, 0.413] |
| Random Forest | 62.6% | 9.5% | 0.375 |
| SVM-RBF | 39.2% | 48.5% | 0.330 |
| XGBoost **with decoy states** | 35.0% | 10.5% | 0.613 |

* **What it detects.** Intercept-resend is detected reliably (F1 0.93), blinding and time-shift
  partly (0.58 / 0.47). PNS, Trojan horse and beam splitting leave Bob's QBER telemetry
  unchanged by design and sit near chance. Decoy states raise PNS F1 from 0.08 to 0.77.
* **Where it fails to generalise.**
  * Unseen distances (80–110 km): FPR rises to 36%.
  * Free-space fading: FPR rises to 50%.
  * Real benign telemetry from a Toshiba system: every block is flagged, because its baseline
    QBER is higher than the simulator's.
* **Comparison with the old numbers.** The earlier 85.6% accuracy (5 classes) came from simulator
  artefacts and row-level cross-validation; [CHANGELOG.md](CHANGELOG.md) explains each artefact.

Results files: `results/metrics/multiattack_model_comparison*.csv`,
`multiattack_per_class_metrics_multiseed*.csv`, `multiattack_channel_generalization.csv`,
`multiattack_heldout_generalization.csv`, `multiattack_latency_vs_intensity.csv`,
`real_vs_sim_qber_calibration.csv`, `real_data_ood_summary.json`.

### Optional: QpiAI Quantum SDK

`qpiai_extras/` (isolated; the pipeline never imports it) adds the following, all with NumPy and
Qiskit fallbacks when no QpiAI account is configured:
* QRNG bits for Alice;
* a circuit-level BB84 / intercept-resend check against the NumPy simulator;
* Bloch-sphere and circuit figures.

Put the key in `QPIAI_API_KEY` or a git-ignored `qcloud.env`. Never commit it. Run
`python -m qpiai_extras.run_demo`.

---

## 1. Scientific Overview & Problem Formulation

Standard BB84 QKD networks rely on a **static Quantum Bit Error Rate (QBER) abort threshold** (typically $e_{\text{abort}} \approx 11\%$, derived from the Shor-Preskill security bound) or scalar statistical bounds (such as Hoeffding's inequality) to flag potential eavesdropping.

However, intelligent adversaries (Eve) can execute **fractional/stealthy intercept-resend (IR) attacks**, intercepting only a proportion $p_{\text{eve}} \in [0.10, 0.30]$ of transmitted photon pulses. Because the induced error $\Delta e \approx \frac{1}{4} p_{\text{eve}} (1 - 2 e_0)$ produces a total QBER of $\approx 5.5\% - 9.5\%$, it remains completely beneath the static $11\%$ threshold, leaking partial key information without triggering a hardware abort.

While modern ML classifiers can reach high static classification accuracy, literature frequently overlooks three critical operational imperatives:
1. **Detection Latency ($\Delta T = T_{\text{detect}} - T_{\text{attack}}$)**: How many elapsed sifted photon blocks occur before the detector issues a true alarm?
2. **Cross-Channel Noise Generalization**: How does a model trained under stationary depolarizing fiber noise ($e_0 = 3\%$) degrade when deployed onto non-stationary free-space/satellite links subject to log-normal atmospheric turbulence and scintillation drift?
3. **Transparent Explainability (XAI)**: Can the detector provide verifiable, feature-level attribution (via SHAP TreeExplainer) justifying an autonomous key-abort decision?

**Phase 3 adds a fourth, and reframes the contribution** (see Section 2): the most dangerous attacks of 2024–2026 are not channel attacks at all. They are *implementation* attacks that violate the device assumptions every security proof takes as a premise. The question becomes not "can ML detect Eve in the key?" but **"which observable changes in device behavior can ML flag during operation?"**

---

## 2. Threat model and protocol scope

QKD security depends on the protocol proof, implementation assumptions, parameter
estimation, authentication and laboratory isolation. A classifier alarm is an
operational warning; a normal prediction is **not a security certificate**.

| Attack surface | BB84 / decoy BB84 | MDI / SNS-TF | This repository |
|---|---|---|---|
| Channel interception | Parameter estimation and privacy amplification under the chosen proof | Protocol-specific estimation remains necessary | Calibrated synthetic benchmarks |
| Multiphoton leakage | Decoy estimation bounds single-photon contributions under source assumptions | Source/decoy assumptions remain necessary | Signal/decoy/vacuum simulations |
| Measurement-device attacks | Trusted-detector assumptions can fail | Untrusted relay allowed under the corresponding proof and its assumptions | Analytic illustrations, not an arbitrary-adversary proof |
| Source leakage / light injection | Characterization, isolation, countermeasures and suitable bounds required | Source characterization and isolation also required | Phenomenological monitor; no leakage-to-key security bound |
| CV-QKD attacks | Outside this discrete-variable model | Outside this model | Not implemented |

The MDI slider is an **ANALYTIC MODEL**: it stipulates a particular Z-measure/resend
strategy and a linear X-error mixture. Its approximately 20% illustrative threshold
is not a universal detector-control threshold. MDI/TF rate calculations are
asymptotic models, not hardware validation. See [limitations](LIMITATIONS.md).

Source attacks can have simultaneous effects: injected light may damage isolation
and seed the laser. A damage-to-Trojan-horse schedule does not imply pure component
effects. The watchdog measures an assumed incident-light tap with spectral
sensitivity. Returned light is hidden truth, not watchdog telemetry.
Modulation-clock correlation and hardware-calibrated source bounds are not implemented.

## 3. Phase 3 implementation and reproduction

Phase 3 contains a stateful source simulator, causal telemetry features, XGBoost,
an independent benign-only Isolation Forest, and a Source Monitor dashboard tab.
Minimal instrumentation uses watchdog power, intensity, temperature and modulator
bias; instrumented mode adds isolation, extinction ratio, SMSR, RIN and wavelength.
All source mappings in `config/source_side.yaml` are versioned **ASSUMPTIONS**.

The hardened evaluator requires a **new alarm during exposure**. It reports
persistent-effect detection, pre-onset false alarms, startup attacks, censoring,
and both chain stages separately. Missed latency is missing. Calibration devices
and operating-point selection devices are disjoint. Temperature-calibrated class
probabilities and validation-normalized classifier/anomaly excursions replace the
prototype's mixed-scale maximum. The fused score is **not** an attack probability.
Engineering limits, EWMA, CUSUM, classifier, anomaly and receiver-only baselines use
the same operating-point procedure; achieved FAR and exposure are reported.

Use Python **3.13.13** and the exact tested environment in
[requirements-phase3-lock.txt](requirements-phase3-lock.txt), including NumPy
**2.1.3**, consistent with `requirements.txt`:

```powershell
python -m venv .venv-phase3
.venv-phase3/Scripts/python -m pip install -r requirements-phase3-lock.txt
.venv-phase3/Scripts/python -m pytest -q
.venv-phase3/Scripts/python reproduce.py --phase3 --quick
.venv-phase3/Scripts/python scripts/13_source_side_pipeline.py --bounded --seeds 42 43 44 --workers 2 --activate-local
# Use the completed run path printed by the pipeline:
.venv-phase3/Scripts/python scripts/14_source_monitor_replay.py results/phase3/runs/<run-id>
.venv-phase3/Scripts/python run_dashboard.py
```

Immutable runs under `results/phase3/runs/<UTC-time>-<id>/` contain manifests,
source snapshots/hashes, package versions, config, schemas and checksums. Each
session has observations, hidden truth and full schedule/device/sensor metadata.
Replay reproduces every event row exactly. Only `--activate-local` selects a
validated completed run for the local dashboard; smoke models cannot be activated.
Generated models and telemetry are excluded from Git and reproduced locally.

The earlier **132/150** result in `results/phase3/full` is a **superseded prototype
result**, preserved unchanged. Bounded audit runs are **exploratory synthetic
evidence**, not frozen final tests or hardware validation. Stress tests and
ablations run across seeds; seeding and photorefraction are separately held out
for unknown-rejection evaluation. Longer benign sessions and seed-then-device
bootstrap intervals improve reporting, but few model seeds and finite exposure
still limit precision. Zero observed false alarms do not imply zero risk.
See [Phase 3 guide](docs/PHASE3.md) and [hardening audit](docs/PHASE3_HARDENING.md)
for measured results, the evidence ledger and remaining limitations.

---

### Latest bounded audit results

Seeds 42–44; 45 nominal attack sessions and 10.2 benign test hours per profile.

| Monitor | New alarm during exposure | False alarms | Detected-only mean latency |
|---|---:|---:|---:|
| Minimal source fusion | 35/45 (77.8%) | 0 | 24.6 s |
| Instrumented source fusion | 36/45 (80.0%) | 1 | 35.6 s |
| Receiver-only baseline | 22/45 (48.9%) | 0 | 83.1 s |

**83 tests passed, 1 skipped** (external real telemetry absent); bounded replay
reproduced **10,458 event rows exactly**. [Run summary](results/phase3/hardened_bounded_summary.csv).
These are exploratory synthetic results. Spectral-blind detection falls to
24.4% / 40.0% for minimal / instrumented profiles. Known-sample false rejection
ranges from 51.7–83.6% / 2.0–66.4% across seeds: the novelty flag needs further
validation and is not operationally reliable. See the audit for uncertainty,
misses, stress tests and the distinction from the superseded 132/150 result.

## 4. Repository Architecture

```text
qkd-explainable-anomaly-detection/
├── config.yaml                  # pipeline + ML settings
├── config/physics.yaml          # every physical parameter, with sources
├── reproduce.py, Makefile       # one-command reproduction
├── requirements.txt             # loose bounds; requirements-lock.txt = exact versions
├── AUDIT.md CHANGELOG.md DATA_CARD.md DATASETS.md LIMITATIONS.md LICENSE CITATION.cff
├── backend/server.py            # FastAPI: simulate (blind/guided/custom), explain, benchmarks, channel, data, mdi
├── frontend/                    # no-build UI: 9 tabs, every chart labelled with source + seed
├── src/
│   ├── config.py                # merges config.yaml + config/physics.yaml; config hash, git commit
│   ├── simulator/
│   │   ├── calibrated.py        # calibrated WCP BB84 (7 classes, decoy option, SPAD/SNSPD, FSO fading)
│   │   ├── bb84_engine.py       # legacy pulse-level engine (scripts 01-05) + Qiskit check
│   │   ├── eve_attacks.py, noise_models.py   # legacy attack and noise models
│   │   ├── tf_qkd.py, mdi_qkd.py             # TF-QKD / MDI-QKD / decoy BB84 key rates, PLOB
│   ├── data/
│   │   ├── dataset.py           # session-seeded generation, session splits, export + sidecars, leakage audit
│   │   └── real_telemetry.py    # Zenodo 21792844 loader (real benign telemetry)
│   ├── features/time_series.py  # causal 63-feature extractor (+ optional decoy group)
│   ├── evaluation/              # finite-size CIs, security metrics, legacy benchmarks
│   └── models/                  # Hoeffding baseline, RF/XGB wrappers, SHAP
├── scripts/
│   ├── 01-05                    # legacy 3-class pipeline and TF-QKD
│   ├── 06_generate_multiattack_dataset.py  # calibrated train / test / held-out (+ --smoke)
│   ├── 07_train_multiattack_classifiers.py # XGBoost, GroupKFold by session
│   ├── 08_evaluate_multiattack.py          # test + held-out, threshold baselines
│   ├── 09_explain_and_ablation.py          # SHAP + feature-group ablation
│   ├── 10_multiseed_evaluation.py          # RF / XGB / SVM, 5 seeds, CIs (+ --decoy)
│   ├── 11_mdi_protocol_comparison.py       # MDI-QKD curves + trial sessions
│   └── 12_real_data_evaluation.py          # real-data calibration + OOD test (+ --download)
├── qpiai_extras/                # optional QpiAI SDK integration (QRNG, circuits, figures)
├── docs/ATTACK_MODELS.md
└── tests/                       # 59 pytest tests, incl. the physics checks
```

---

## 5. Quickstart Guide

See [Section 0](#0-current-results-and-one-command-reproduction) for `reproduce.py`. To run
single steps:

```bash
python scripts/06_generate_multiattack_dataset.py   # calibrated dataset (--smoke for a tiny one)
python scripts/07_train_multiattack_classifiers.py  # XGBoost checkpoint for the dashboard
python scripts/10_multiseed_evaluation.py --seeds 42 43 44 45 46 --sessions-per-class 200 [--decoy]
python scripts/12_real_data_evaluation.py [--download]
python -m pytest
```

Approximate runtimes on a 24-core desktop:

| Script | Runtime |
|---|---|
| 06 | ≈ 100 s |
| 07 | ≈ 70 s |
| 09 | ≈ 80 s |
| 10 | ≈ 75 min (SVM-dominated; ≈ 47 min with `--decoy`) |
| 12 | ≈ 1 min |

---

## 6. Phase 1 Protocol Classes & Threat Model

| Class | Label | Eavesdropping Strategy | Interception Rate $p_{\text{eve}}$ | Expected QBER ($e_0 = 3\%$) |
|---|:---:|---|:---:|:---:|
| **Class 0** | `0` | **Benign Noise** (Stationary fiber or scintillation) | $0.0$ | $\approx 3.0\%$ |
| **Class 1** | `1` | **Full Intercept-Resend (IR)** | $1.0$ | $\approx 26.5\%$ (Exceeds $11\%$ abort) |
| **Class 2** | `2` | **Fractional/Stealthy IR** | $[0.10, 0.30]$ | $\approx 5.3\% - 9.9\%$ (**Sub-threshold stealth**) |

---

## 7. Phase 1 Results (reproduced 2026-10-26)

All numbers below come from running scripts 01-05 with `random_seed: 42` (Python 3.13, numpy 2.1, scikit-learn 1.9, xgboost 3.4).

### 7.1 Physics verification (`01_verify_physics.py`)

| Check | Empirical | Theory |
|---|:---:|:---:|
| Noiseless, no Eve | QBER 0.0000 | 0 |
| Full IR, $e_0=0$ | 0.2533 | 0.2500 |
| Fractional IR, $e_0=3\%$, $p_{\text{eve}}=0.2$ | 0.0780 | 0.0770 |
| Free-space drift: QBER std / lag-1 autocorrelation | 0.0063 / 0.70 | ≈0.009 / 0.75 |
| Qiskit measure-and-resend, $p_{\text{eve}}=1$ | 0.2446 | 0.2500 |

### 7.2 Fiber test set, binary detection (`03_run_benchmarks.py`)

| Detector | Accuracy | F1 | FPR | AUROC |
|---|:---:|:---:|:---:|:---:|
| Static 11% | 0.8318 | 0.6825 | 0.0000 | 0.9982 |
| Hoeffding (single block) | 0.8502 | 0.7268 | 0.0000 | 0.9982 |
| Hoeffding (windowed W=10) | 0.9904 | 0.9861 | 0.0000 | 0.9993 |
| Random Forest (calibrated) | 0.9993 | 0.9990 | 0.0003 | 0.9999 |
| XGBoost (calibrated) | 0.9993 | 0.9990 | 0.0003 | 1.0000 |

Full-IR blocks are trivially detectable and make up half of all attack blocks, so the **stealth subset** (benign vs. fractional IR only, `stealth_subset_benchmarks_fiber.csv`) is the more honest view:

| Detector | Recall on stealth attack | F1 | FPR |
|---|:---:|:---:|:---:|
| Static 11% | 0.036 | 0.069 | 0.0000 |
| Hoeffding (single block) | 0.142 | 0.248 | 0.0000 |
| Hoeffding (windowed W=10) | 0.948 | 0.973 | 0.0000 |
| Random Forest | 0.997 | 0.998 | 0.0003 |
| XGBoost | 0.997 | 0.998 | 0.0003 |

### 7.3 Cross-channel generalization (trained on fiber, tested on free-space)

| Detector | Fiber F1 | Free-space F1 | Free-space FPR |
|---|:---:|:---:|:---:|
| Hoeffding (windowed W=10) | 0.986 | 0.971 | 0.59% |
| Random Forest | 0.999 | 0.901 | **11.6%** |
| XGBoost | 0.999 | 0.931 | **7.7%** |

ML detectors trained on stationary fiber noise mistake atmospheric drift for an attack. Their FPR rises far above the 1% target, while the analytic Hoeffding test degrades gracefully. Multi-class models (`multiclass_benchmarks.csv`) drop from 0.999 to 0.92-0.94 accuracy, and the fractional-IR F1 falls to 0.82 (RF) / 0.86 (XGB).

### 7.4 Detection latency ΔT (mean blocks after onset, 40 sessions per point)

| $p_{\text{eve}}$ | Static 11% | Hoeffding single | Hoeffding windowed | RF | XGBoost |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.05 | never | never | 99.7 (2.5% detected) | 8.9 | 13.0 |
| 0.10 | never | never | 9.5 | 1.7 | 1.4 |
| 0.20 | 78.6 (40% detected) | 23.7 | 4.1 | 0.08 | 0.08 |
| 0.30 | 2.3 | 0.7 | 2.5 | 0.0 | 0.0 |

---

## 8. Comparison with the originally committed results

The scripts were first re-run unchanged (apart from a crash fix) and compared with the committed `results/metrics`:

* **Simulation and all threshold baselines reproduced bit for bit.** Static 11% and both Hoeffding variants gave identical confusion matrices and latencies.
* **ML models differed in the 4th decimal.** For example, Random Forest false positives went from 17 to 20 out of 74,600 benign blocks, and latency at $p_{\text{eve}}=0.05$ went from 9.65 to 8.90 blocks (RF) and from 14.03 to 13.00 (XGBoost). This comes from library-version differences (xgboost 3.4, scikit-learn 1.9).
* **SHAP ranking unchanged** (rolling_mean > qber_current > shannon_entropy > rolling_std > burstiness_index); magnitudes within 0.004.

Issues found and fixed during the review:

| # | Issue | Effect | Fix |
|---|---|---|---|
| 1 | `ml_classifiers.py` used `Path` without importing it | Scripts 03 and 04 crashed with `NameError`; the committed results could not be regenerated | Added the import |
| 2 | The Qiskit check never measured Eve: her H gate just stacked with Bob's | The check landed on ~25% by coincidence, not physics | Real measure, collapse and re-prepare in two circuits, plus a zero-QBER no-Eve check |
| 3 | `scintillation_std: 0.008` gave a free-space QBER std of 0.0002, far below the 0.0075 binomial noise of a block | The cross-channel experiment was vacuous (F1 drop ≈ 0.0003), contradicting the claim above | Raised to 0.30 (QBER 3% ± 0.6-0.9%); the degradation in §7.3 is now real |
| 4 | Physics Test 3 seeded the noise model and Eve with the same seed | Noise flips and interceptions were correlated (identical RNG streams) | Distinct seeds. The dataset pipeline was not affected |
| 5 | Each latency detector consumed a different part of the simulator's random stream | Detectors were compared on different sessions | A fresh, identically seeded simulator for each detector |
| 6 | Python loop over 530k windows | Dataset generation took 9.5 min | Vectorized extractor, matching the reference to within 2e-19: **≈ 70 s** |
| 7 | Multi-class models were trained but never evaluated | — | Added `multiclass_benchmarks.csv` and the stealth-subset table |

---

## 9. Modern Protocol: Sending-or-Not-Sending Twin-Field QKD

BB84 key rates fall linearly with channel transmittance η. No point-to-point protocol can beat the **PLOB bound** $-\log_2(1-\eta)$. **Twin-Field QKD** (Lucamarini et al., *Nature* 2018) interferes weak pulses from Alice and Bob at an untrusted relay (Charlie). Only one photon has to survive *half* of the channel, so the rate scales as $\sqrt{\eta}$. The **sending-or-not-sending** variant (Wang, Yu & Hu, *PRA* 2018) underlies record demonstrations such as ~1000 km of fiber (Liu et al., *PRL* 2023). Because Charlie is untrusted, the protocol is also measurement-device-independent, which removes detector side channels — see the coverage matrix in Section 2.2.

`src/simulator/tf_qkd.py` implements:
* **Z (signal) window:** each side sends a phase-randomized coherent state $\mu_z$ with probability ε, and vacuum otherwise. An effective event is exactly one of Charlie's two detectors clicking.
* **X (decoy) window:** intensities {0, μ₁, μ₂}. Phase post-selection over M = 16 slices measures the phase error.
* **Key rate:** decoy estimation of the single-photon yield $s_1^L$ and the phase-error bound $e_1^{ph,U}$. Asymptotic rate $R = 2\varepsilon(1-\varepsilon)\mu_z e^{-\mu_z} s_1[1-H(e_1^{ph})] - f\,S_Z H(E_Z)$, optimized over (ε, μ_z, μ₁, μ₂).
* **Comparison curves:** decoy-state BB84 (GLLP, infinite decoys) and the PLOB bound, all with the same fiber (0.2 dB/km) and detectors (η_d = 0.6, p_d = 1e-8).
* **Finite-block time series:** binomially sampled counts with an AR(1) drift of the residual phase noise.
* **Phase-tampering attack:** Eve scrambles the phase of a fraction $p_{\text{eve}}$ of pulses. This is a phenomenological model of measure-and-resend on the phase reference.

**Key rates** (`fig6_tfqkd_key_rate_vs_distance.png`, `tfqkd_key_rate_vs_distance.csv`):

| Distance | SNS-TF-QKD | Decoy BB84 | PLOB |
|:---:|:---:|:---:|:---:|
| 100 km | 4.9e-4 | 1.3e-3 | 1.5e-2 |
| 200 km | 4.7e-5 | 1.3e-5 | 1.4e-4 |
| 300 km | 4.6e-6 | 6.2e-8 | 1.4e-6 |
| 400 km | 3.8e-7 | 0 | 1.4e-8 |

SNS-TF-QKD overtakes decoy BB84 at ~150 km and **beats the PLOB bound from ~250 km**. Its maximum distance is 470 km, against 310 km for BB84 with these parameters. Each extra 100 km costs SNS-TF a factor ≈10 ($\sqrt{\eta}$) and BB84 a factor ≈100.

**Attack detection at 200 km** (`fig7_tfqkd_phase_attack_trace.png`, `tfqkd_detection_latency.csv`). Cells show the alarm rate on attacked blocks; ΔT is the mean latency in blocks.

| $p_{\text{eve}}$ | Z-QBER monitor | Static 11% on $E_X$ | Hoeffding on $E_X$ | RF on $E_X$ features |
|:---:|:---:|:---:|:---:|:---:|
| benign FPR | 1.0-1.3% | 0% | **13-17%** | 0.4-0.9% |
| 0.02 | 1.2% (blind) | 0% | 64% | 17% (ΔT 33) |
| 0.05 | 0.8% (blind) | 0% | 97% | 88% (ΔT 7.4) |
| 0.10 | 1.4% (blind) | 0.3% | 99% | 99.8% (ΔT 0.2) |

Phase tampering leaves the Z-window bit errors **exactly** unchanged, because Z pulses are already phase-randomized. A conventional QBER monitor therefore fires only at its own false-alarm rate. The attack shows up only in the X-window phase error, and the slow phase drift makes that error non-stationary. As a result, a Hoeffding test that assumes a fixed baseline false-alarms on 13-17% of benign blocks. The temporal Random Forest, trained on drifting benign data, keeps the FPR below 1% and detects $p_{\text{eve}} \ge 0.05$ within about 7 blocks.

---

## 10. Phase 2: Multi-Attack Classification & the SOC Dashboard
*Reference Architecture:* *"Beyond the QBER Threshold: A Temporal QBER Based Machine Learning Framework for Multi Attack Detection in BB84 QKD"*

Phase 2 scales the framework from binary anomaly detection to a **5-class multi-attack classification engine** powered by XGBoost, with a real-time **Security Operations Center dashboard** (FastAPI + an animated no-build frontend).

### 10.1 The 5 Operational Regimes

| Class | Threat Archetype | Physical Layer Exploited | Primary Feature Fingerprint |
|:---:|---|---|---|
| **0** | **Normal Channel** | Baseline fiber/atmospheric link | Stationary QBER ($\approx 3\%$), symmetric basis errors ($Q_Z \approx Q_X$). |
| **1** | **Fractional IR** | Quantum state vector disturbance | Elevated **jump energy, temporal variance, Fano factor** ($QBER \approx 5.5\% - 9.5\%$). |
| **2** | **PNS Attack** | Laser multi-photon pulse splitting | **Zero added QBER ($\Delta QBER \approx 0$)**, abnormal drop in key generation rate and transmission efficiency. |
| **3** | **Fake State** | SPAD detector blinding into linear mode | **Near-zero QBER ($\approx 0.3\%$)**, deterministic basis-coincidence, abnormal error-gap CV. |
| **4** | **Time-Shift** | Timing & efficiency detector mismatch | Severe **basis asymmetry ($\|Q_{\text{rect}} - Q_{\text{diag}}\| \gg 0$)** beneath 11% average threshold. |

### 10.2 63 Physics-Informed Feature Categories ($W = 50$)
Extracted over causal backward-looking sliding windows ($W=50$) with zero future leakage:
1. **Statistical (7):** `qber_mean`, `qber_variance`, `qber_std`, `qber_skewness`, `qber_kurtosis`, `qber_iqr`, `qber_cv`.
2. **Burst & Instability (10):** `spike_density`, `burst_duration`, `jump_energy`, `temporal_drift_slope`, `max_jump`, `jump_count`, `error_gap_mean`, `error_gap_std`, `error_gap_cv`, `burstiness_index`.
3. **Spectral & Temporal (12):** `autocorr_lag_1` to `lag_5`, `dft_low_energy`, `dft_mid_energy`, `dft_high_energy`, `spectral_entropy`, `spectral_centroid`, `shannon_entropy`, `ema_divergence`.
4. **Basis-Dependent (12):** `qber_rectilinear` ($q_Z$), `qber_diagonal` ($q_X$), `basis_asymmetry`, `basis_ratio`, `basis_burst_energy_z`, `basis_burst_energy_x`, `basis_variance_ratio`, `basis_cross_correlation`, `basis_skewness_diff`, `basis_kurtosis_diff`, `sift_ratio_z`, `sift_ratio_x`.
5. **Channel Interaction & Multi-Scale (22):** `key_generation_rate`, `transmission_efficiency`, `qber_loss_interaction`, `qber_loss_product`, `loss_anomaly_index`, `dark_count_ratio`, nested multi-scale statistics (`W=10, 25, 50` mean, std, fano, drift), `error_clustering_coefficient`, `loss_gradient`, `efficiency_entropy`, `qber_stability_index`.

### 10.3 Experimental Benchmarks (superseded; see Section 0)

> The numbers in 10.3–10.5 are from the 5-class legacy engine and are kept for the record.
> The audit ([AUDIT.md](AUDIT.md)) found further artefacts in that engine: every class except
> PNS treated pulses as perfect single photons, benign channels were stationary, and CV was
> row-level. Current results are in Section 0 and CHANGELOG.md.

The originally committed Phase 2 figure was 99.93% accuracy. **That number was an artefact of the simulator, not a generalization estimate**, and it has been corrected. Section 10.5 documents exactly what was wrong and how it was fixed.

Headline result (`scripts/10_multiseed_evaluation.py`, 5 seeds x 2,000 sessions each, session-level 80/20 split so that train and test never share a session):

| Detector | Accuracy | Macro-F1 | FNR (attack vs normal) | FPR |
|---|:---:|:---:|:---:|:---:|
| **Conventional Fixed 11% QBER Threshold** | - | - | **99.38% ± 0.31%** | 0.00% |
| **Proposed XGBoost Multi-Attack Framework** | **85.63% ± 1.36%** | **0.8470 ± 0.0144** | **11.53% ± 1.11%** | 7.68% ± 1.53% |

Single-run confirmation on the full canonical dataset (1,000 sessions/class, scripts 06-08): accuracy 86.29%, macro-F1 0.8556, FNR 11.94%, against 40.81% / 0.2927 / 99.64% for the fixed threshold.

The headline claim survives the correction, and is the stronger one for being honest: **the fixed 11% threshold misses 99.4% of attack blocks while the temporal model misses 11.5%** - roughly a nine-fold reduction in false negatives, with no class trivially identifiable.

#### Per-class F1 (mean ± std over 5 seeds)

| Class | F1 |
|---|:---:|
| Normal Channel | 0.8826 ± 0.0104 |
| Fractional IR | 0.8746 ± 0.0283 |
| PNS Attack | 0.8021 ± 0.0235 |
| Fake State | 0.7417 ± 0.0113 |
| Time-Shift | 0.9339 ± 0.0181 |

Fake State is now the hardest class, which is physically right: blinding is *designed* to hold the error rate near baseline, and once benign sessions may run at QBER as low as 0.5% it no longer stands out. Time-Shift stays easiest, because its basis asymmetry is a genuine two-channel signature rather than a simulator artefact.

#### Leave-one-group-out feature ablation (`fig_ablation_5class.png`)

| Group removed | Ablated accuracy | Drop |
|---|:---:|:---:|
| Basis-Dependent (12) | 79.12% | **-7.16%** |
| Channel Interaction (22) | 80.93% | **-5.36%** |
| Burst & Instability (10) | 85.89% | -0.39% |
| Spectral & Temporal (12) | 85.91% | -0.37% |
| Statistical (7) | 85.96% | -0.33% |

With the leakage removed this ablation is far more informative than before, when every group scored a drop under 1%. **Basis-dependent features are now the most important group**, which matches the physics: separating Time-Shift and Fake State from a merely lossy benign channel requires comparing the two bases against each other, not tracking scalar QBER.

### 10.4 Running the Security Operations Center dashboard

```bash
python run_dashboard.py            # http://127.0.0.1:8000
python run_dashboard.py --reload   # auto-reload while editing
```

The dashboard is a **FastAPI backend plus a no-build static frontend** - no Streamlit, no npm, no bundler. `backend/server.py` exposes the simulator, the trained model, SHAP attribution and the MDI-QKD comparison as JSON; `frontend/` is plain HTML, CSS and JavaScript with hand-written SVG charts and a canvas animation.

| Endpoint | Purpose |
|---|---|
| `GET /api/meta` | class metadata, scenarios, feature groups, nuisance ranges, headline metrics |
| `GET /api/trial/{cls}` | replay a pre-generated trial session (instant) |
| `POST /api/simulate` | run a fresh BB84 session with chosen channel, length, onset and seed |
| `GET /api/explain` | SHAP attribution for one block of a session |
| `GET /api/mdi` | detector-attack sweep and key-rate curves |
| `GET /api/benchmarks` | the corrected metric tables |

**Six views:**
* **Start Here** - the problem in plain language, the corrected headline numbers, and the protocol decision table.
* **Live Monitor** - six scenario presets, each chosen to expose a different blind spot of the 11% rule (clean baseline, sub-threshold stealth, zero-QBER PNS, detector blinding where the QBER *drops*, time-shift basis split, and an attack on a drifting free-space link). An animated canvas shows photons travelling Alice to Bob with polarization glyphs; when the attack is live, Eve's node pulses and intercepted photons change colour. Block-by-block playback with a scrubber drives the charts, the KPI tiles and the classifier belief bars together.
* **Why? (SHAP)** - per-block diverging attribution, group-level roll-up, plain-language reading, and an explicit flag when the block was misclassified.
* **MDI-QKD** - a slider for how much of the relay Eve controls, with BB84 and MDI-QKD side by side, plus the monitor and key-rate curves.
* **Benchmarks** - corrected metrics, per-seed stability, the leakage story, per-class tables and the ablation.
* **Threat Atlas** - per-class physics, the Phase 3 source-side attacks, the 63 features, the nuisance ranges and the limitations.

Motion is used to carry meaning rather than decorate: chart paths draw in, the live point pulses, probability bars animate between blocks, and the monitor panel takes on a pulsing red border the moment the classifier leaves the Normal class. Everything collapses to a static render under `prefers-reduced-motion`.

Trial sessions come from `scripts/11_mdi_protocol_comparison.py` and carry per-block predictions, so replay costs nothing. Live simulation passes `randomize_nuisance=True`, so it draws from the same leakage-free distribution the model was trained on.

### 10.5 What the label leakage was, and how it was fixed (task P2.1)

**The defect.** `simulate_block_vectorized` hard-coded `base_channel_transmittance = 1.0` and `dark_count_rate = 1e-5` for every class, and each attack carried fixed strength constants. As a result `transmission_efficiency` collapsed onto a per-class constant:

| Class | mean | std | min | max |
|---|:---:|:---:|:---:|:---:|
| 0 Normal | 1.000000 | **0.000000** | 1.000000 | 1.000000 |
| 1 Fractional IR | 1.000000 | **0.000000** | 1.000000 | 1.000000 |
| 2 PNS | 0.407803 | 0.291112 | 0.075605 | 0.983477 |
| 3 Fake State | 0.646346 | 0.173570 | 0.470078 | 0.990215 |
| 4 Time-Shift | 0.867116 | 0.068717 | 0.750840 | 0.996992 |

A feature with **zero variance** across two entire classes is a perfect separator, and `key_generation_rate`, `loss_anomaly_index` and the `qber_loss_*` features all derive from it.

**The fix.** `src/simulator/bb84_engine.py` now defines `NUISANCE_RANGES` and a `randomize_nuisance` flag. Each session draws its own baseline QBER, channel transmittance, dark-count rate, per-basis detector efficiency and per-attack strength parameters, from ranges that deliberately overlap so that a benign session can look like a weak attack:

* benign base QBER down to 0.5%, overlapping the fake-state blinding residual (0.2-2.0%)
* benign channel loss spanning 10-95%, overlapping PNS yield suppression
* benign detectors carrying up to 30% basis-efficiency mismatch - every real receiver has some - overlapping weak Time-Shift
* IR interception down to 5%, where the induced error is comparable to benign drift

The same feature afterwards, overlapping across all five classes:

| Class | mean | std | min | max |
|---|:---:|:---:|:---:|:---:|
| 0 Normal | 0.4306 | 0.2156 | 0.0924 | 0.8654 |
| 1 Fractional IR | 0.4103 | 0.2246 | 0.1090 | 0.8525 |
| 2 PNS | 0.1684 | 0.1360 | 0.0216 | 0.7555 |
| 3 Fake State | 0.2534 | 0.1519 | 0.0405 | 0.7911 |
| 4 Time-Shift | 0.4259 | 0.2121 | 0.0963 | 0.8441 |

**Verification.** `scripts/10_multiseed_evaluation.py` writes `multiattack_leakage_audit.json`, recording the best single-feature one-vs-rest AUC per class. It fell from 0.97-0.998 before the fix to a mean maximum of **0.932 ± 0.014** after. Some separability is expected and legitimate - these are physically distinct attacks - but no class can be identified from a single feature any more. `tests/test_multiattack.py::test_nuisance_randomization_breaks_per_class_fingerprints` asserts the overlap, so the defect cannot silently return.

**Note on the cross-validation figure.** `cv_multiattack_xgboost.json` reports 90.55% ± 0.15%, above the 86.29% held-out result, because that CV shuffles at *block* level and therefore splits overlapping windows from the same session across folds. The session-level held-out number is the one to quote.

---

## 11. Mathematical Formulations

### Hoeffding Baseline Threshold (Lee et al., 2022)
For a sifted key block of size $N$ with baseline error $e_0$ and significance level $\alpha = 0.01$:
$$\hat{e}_{\text{thresh}} = e_0 + \sqrt{\frac{\ln(1/\alpha)}{2N}}$$

### Causal Feature Extraction
Given discrete error sequence $E = [e_1, e_2, \dots, e_t]$, feature vectors $x_t \in \mathbb{R}^d$ are computed strictly using backward sliding windows $E_t = [e_{t-W+1}, \dots, e_t]$ to ensure zero future data leakage. Extracted features include current QBER, rolling mean and standard deviation, Fano factor, first-order drift slope, EMA divergence, rolling skewness, kurtosis, Shannon entropy, burstiness index and lag-1 autocorrelation.

---

## 12. Limitations

See [LIMITATIONS.md](LIMITATIONS.md). In short:
* the data is simulated;
* the attack models are simplified;
* thresholds and key rates are asymptotic (per-block Wilson intervals are reported);
* nothing has been validated on hardware;
* the model does not transfer to the real telemetry tested.

---

## 13. Citation

If you use this benchmark codebase in your research, please cite it (see also [CITATION.cff](CITATION.cff)); if you use the real telemetry, cite Zenodo 21792844 as well:
```bibtex
@inproceedings{qkd_explainable_temporal_ml_2026,
  title={Explainable ML for Eavesdropping Detection in Quantum-Key Distribution (BB84)},
  author={Kompi, Samarth},
  booktitle={Undergraduate Experiential Learning Conference in Quantum Engineering and Computing},
  year={2026}
}
```
