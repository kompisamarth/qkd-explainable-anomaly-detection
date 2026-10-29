# Explainable ML for Eavesdropping Detection and Device-Assumption Monitoring in QKD

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE Format](https://img.shields.io/badge/Paper-IEEE%20EL%20Conference-orange.svg)](#)

A research pipeline and benchmark implementation for explainable, low-latency machine learning detection of stealthy attacks on Quantum Key Distribution. It covers the traditional **BB84** protocol, a **Sending-or-Not-Sending Twin-Field QKD (SNS-TF-QKD)** simulator, and — from Phase 3 — a **source-side side-channel monitor** targeting the transmitter light-injection attacks that no security proof currently covers.

**Project status:** Phase 1 and Phase 2 complete (Sections 7–10). Phase 3 is specified in [Section 3](#3-phase-3-roadmap) and not yet implemented.

---

## 1. Scientific Overview & Problem Formulation

Standard BB84 QKD networks rely on a **static Quantum Bit Error Rate (QBER) abort threshold** (typically $e_{\text{abort}} \approx 11\%$, derived from the Shor-Preskill security bound) or scalar statistical bounds (such as Hoeffding's inequality) to flag potential eavesdropping.

However, intelligent adversaries (Eve) can execute **fractional/stealthy intercept-resend (IR) attacks**, intercepting only a proportion $p_{\text{eve}} \in [0.10, 0.30]$ of transmitted photon pulses. Because the induced error $\Delta e \approx \frac{1}{4} p_{\text{eve}} (1 - 2 e_0)$ produces a total QBER of $\approx 5.5\% - 9.5\%$, it remains completely beneath the static $11\%$ threshold, leaking partial key information without triggering a hardware abort.

While modern ML classifiers can reach high static classification accuracy, literature frequently overlooks three critical operational imperatives:
1. **Detection Latency ($\Delta T = T_{\text{detect}} - T_{\text{attack}}$)**: How many elapsed sifted photon blocks occur before the detector issues a true alarm?
2. **Cross-Channel Noise Generalization**: How does a model trained under stationary depolarizing fiber noise ($e_0 = 3\%$) degrade when deployed onto non-stationary free-space/satellite links subject to log-normal atmospheric turbulence and scintillation drift?
3. **Transparent Explainability (XAI)**: Can the detector provide verifiable, feature-level attribution (via SHAP TreeExplainer) justifying an autonomous key-abort decision?

**Phase 3 adds a fourth, and reframes the contribution** (see Section 2): the most dangerous attacks of 2024–2026 are not channel attacks at all. They are *implementation* attacks that violate the device assumptions every security proof takes as a premise. The question becomes not "can ML detect Eve in the key?" but **"can ML certify at runtime that the security proof's assumptions still hold?"**

---

## 2. Threat Landscape (2024–2026) and Protocol Selection

### 2.1 Attack taxonomy by structural location

The practically hardest-to-manage attacks in the current literature sort into three structural locations. What matters for protocol selection is **where** the attack lives, not how it is implemented.

| Group | Attacks | Why it is hard |
|---|---|---|
| **(A) Measurement / detector side** | Detector blinding & faked-state, detector-efficiency mismatch, time-shift, backflash leakage | Blinding has been **experimentally demonstrated to recover a full key without raising QBER**, on both APD and superconducting-nanowire detectors. The one obvious alarm bell never rings. Backflash leaks up to ~10% of detection events back down the channel. |
| **(B) Source / transmitter side (light injection)** | Trojan-horse (THA), laser seeding / injection locking, induced photorefraction, laser damage | Eve injects light **into** Alice's device. The attack surface is the full spectrum (visible through ~1500–2100 nm), not just 1550 nm. Several of these — notably light injection and induced photorefraction — are **not covered by existing security proofs**. |
| **(C) CV-QKD specific** | Homodyne-detector blinding, local-oscillator and calibration attacks | Different physical layer (quadratures, excess noise). Out of scope for this DV-focused project; noted for completeness. |

### 2.2 Protocol coverage matrix

"Closed" means the attack is **structurally impossible or harmless by construction**, not merely detectable by a statistic.

| Attack | Location | BB84 | Decoy-BB84 | MDI-QKD | SNS-TF-QKD | DI-QKD |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Intercept-resend | channel | detectable | detectable | detectable | detectable | closed |
| Photon-number splitting | source/channel | **open** | closed | closed | closed | closed |
| Beam splitting | channel | open | partial | closed | closed | closed |
| **Detector blinding / faked-state** | detector | **open** | **open** | **closed** | **closed** | closed |
| **Detector-efficiency mismatch** | detector | **open** | **open** | **closed** | **closed** | closed |
| **Time-shift** | detector | **open** | **open** | **closed** | **closed** | closed |
| **Backflash leakage** | detector | **open** | **open** | **closed** | **closed** | closed |
| **Trojan-horse (THA)** | **source** | open | open | **OPEN** | **OPEN** | **OPEN** |
| **Laser seeding / injection locking** | **source** | open | open | **OPEN** | **OPEN** | **OPEN** |
| **Induced photorefraction** | **source** | open | open | **OPEN** | **OPEN** | **OPEN** |
| **Laser damage** | **source** | open | open | **OPEN** | **OPEN** | **OPEN** |

### 2.3 Decision: MDI family (SNS-TF-QKD) as backbone

**The most efficient protocol against this threat list is the measurement-device-independent family — concretely SNS-TF-QKD, which this repository already implements.**

The reasoning:

* **It eliminates the entire detector-side group (A) by construction.** In MDI-QKD and TF-QKD the measurement device is operated by an *untrusted* relay (Charlie). Eve is *permitted* to control the detectors completely; doing so cannot bias the key, only reduce the rate. Blinding, faked-state, efficiency mismatch, time-shift and backflash all stop being attacks. For blinding in particular — where a statistical monitor can only ever be a heuristic, because the attack produces no QBER signature — this is the difference between a guess and a proof.
* **It is deployable today.** Metropolitan MDI-QKD networks are operational, and SNS-TF-QKD holds the ~1000 km fiber record (Liu et al., *PRL* 130, 210801, 2023). DI-QKD closes marginally more in theory but needs detection-loophole-free Bell violation; its rates and distances remain impractical.
* **It costs this project almost nothing.** `src/simulator/tf_qkd.py` already models the untrusted relay, beam-splitter interference, decoy-state single-photon yield and phase-error estimation. MDI-QKD is the sibling protocol.

### 2.4 Why no protocol closes group (B) — the sealed-laboratory assumption

This is the pivot that reshapes the methodology.

> **Every** QKD security proof — BB84, decoy-state, MDI, and even device-independent — assumes that *no unwanted information leaves Alice's and Bob's laboratories*. Trojan-horse and the wider light-injection family attack that premise **from outside the protocol**. They are therefore not fixable by any choice of protocol, including DI-QKD.

Group (B) can only be addressed by hardware countermeasures (optical isolators, spectral filters, power limiters, watchdog photodiodes) **plus runtime monitoring** — and the literature openly concedes that the proofs have not caught up with the attacks.

**That gap is the Phase 3 research contribution.** The protocol closes what proofs can close; the ML layer monitors what proofs cannot.

### 2.5 Why temporal ML is the right instrument for group (B)

This is not an arbitrary pairing. The group (B) attacks have **inherently temporal signatures**, which is precisely what this project's existing causal feature machinery (63 features, drift slope, multi-scale statistics, autocorrelation, spectral entropy) was built to capture:

| Attack | Physical mechanism | Temporal signature |
|---|---|---|
| **Induced photorefraction** | Short-wavelength light builds a space-charge field in Alice's LiNbO₃ modulator, shifting its photorefractive properties | **Slow saturating-exponential drift** of modulator bias and extinction ratio over minutes–hours; partially recovers in the dark. Invisible to any static threshold; trivial for a drift-slope feature. |
| **Laser damage** | High-power laser permanently degrades the isolator/attenuator | **Monotonic, irreversible** step/ramp in optical isolation. Distinguished from photorefraction precisely by *non-recovery*. |
| **Laser seeding / injection locking** | Injected light seeds Alice's laser cavity | **Abrupt step** in side-mode suppression ratio, relative intensity noise and emitted wavelength at attack onset. |
| **Trojan-horse** | Eve probes and reads back-reflections | Back-reflection power at the watchdog photodiode, **phase-locked to Alice's modulation frame** — Eve must probe *during* state preparation, so the cross-correlation with the modulation clock is the signature. |

Note also that **laser damage is a precursor**: degrading isolation is what *enables* THA and laser seeding. Phase 3 therefore models attack **chains**, not just isolated attacks, and targets detection of the precursor before the downstream attack succeeds.

### 2.6 Simulated verification of that decision (`src/simulator/mdi_qkd.py`)

The claim in 2.3 is now implemented and measured rather than asserted. Charlie's station is modelled from first principles - a 50:50 beam splitter, a polarising beam splitter per output arm and four detectors - with gains and error rates obtained by averaging the two-fold coincidence probabilities over the random relative phase of two independent lasers, the same method already used in `tf_qkd.py`.

**Physics checks** (`tests/test_mdi_qkd.py`, 8 tests):

| Check | Result |
|---|---|
| Ideal relay, no dark counts or misalignment | `E_Z` exactly 0 |
| Single-photon phase error vs theory | `e11` = 0.0296 = 2·e_d(1−e_d), to 1e-12 |
| Rate scaling | falls ~100x per 100 km (linear in η_a·η_b) |
| Does MDI-QKD beat the PLOB bound? | No, at any distance - as it should not |
| SNS-TF-QKD vs MDI-QKD at 300 km | TF-QKD wins, confirming its √η advantage |

**The detector attack, run against both protocols** (`fig9_detector_attack_comparison.png`, `mdi_detector_attack_comparison.csv`):

| Eve controls the measurement device | BB84 + blinding | MDI-QKD + Eve as Charlie |
|---|:---:|:---:|
| Key bits Eve learns | **100%** | 100% in the Z basis |
| Error rate the operator observes | 0.50% | 0.50% |
| Independent security monitor | none - Eve forges the check basis too | **X-basis phase error, 50%** |
| Protocol aborts? | **No** | **Yes** |

MDI-QKD trips its monitor once Eve controls just **20%** of the relay; BB84 never trips at any control fraction. The reason is structural: measuring the pulses to learn Z-basis bits destroys their X-basis coherence, and that phase error is computed by Alice and Bob from states *they* prepared, so Eve cannot forge it.

**The price is rate, not security.** MDI-QKD's key rate scales with the product of the two arm transmittances, so it is slower than BB84 at every distance and never beats the PLOB bound. That is TF-QKD's trick, not MDI's.

| Distance | MDI-QKD | SNS-TF-QKD | Decoy BB84 |
|:---:|:---:|:---:|:---:|
| 100 km | 9.3e-5 | 4.9e-4 | 1.3e-3 |
| 200 km | 9.1e-7 | 4.7e-5 | 1.3e-5 |
| 300 km | 8.8e-9 | 4.6e-6 | 6.2e-8 |

---

## 3. Phase 3 Roadmap

Phase 3 is the final submission phase. It has two halves: hardening what Phase 2 already claims, then building the source-side monitor.

### 3.0 Phase 2 close-out (prerequisite — do first)

The Phase 2 classifier reports 99.93% accuracy (Section 10.3). That number is **not trustworthy as a generalization estimate**, and must be corrected before it goes into a submission.

| # | Task | Rationale |
|---|---|---|
| P2.1 | **DONE** - **Overlap the nuisance ranges across classes.** Draw per-session baseline QBER, detector efficiency, channel loss and misalignment from *overlapping* distributions instead of per-class constants. | Each attack class currently carries a hand-coded, near-deterministic fingerprint (fake-state forces effective noise to `min(0.005, ·)`; time-shift hard-codes η_Z = 1.0 vs η_X = 0.4). The model is partly learning the simulator, not the physics. |
| P2.2 | **DONE** - **Re-report accuracy as mean ± std over ≥ 5 seeds.** Accuracy fell from 99.93% to 85.63% ± 1.36% (Section 10.3). | A single-seed 99.93% on synthetic multi-class data is a red flag reviewers will press on. |
| P2.3 | **Add detection latency ΔT and FPR to the 5-class evaluation.** | Currently only accuracy/F1 are reported for Phase 2, losing the operational framing that makes Phase 1 strong. |
| P2.4 | **Run the 5-class model cross-channel (fiber → free-space).** | Section 9.3 is the most intellectually honest result in the repo; Phase 2 has no equivalent. |

**Acceptance:** no single feature separates any class pair; reported accuracy carries a seed-variance band; ΔT and FPR tables exist for all five classes.

### 3.1 WS-1 — MDI-QKD backbone  *(core model DONE — see Section 2.6)*

Extend `src/simulator/tf_qkd.py` into a shared MDI module covering both MDI-QKD (Bell-state measurement at Charlie, decoy states, Z/X encoding) and the existing SNS-TF-QKD.

**Deliverables:** key rate vs. distance curves for MDI-QKD alongside the existing SNS-TF and decoy-BB84 curves; a demonstration that a simulated blinding/time-shift attack on Charlie's detectors yields Eve **no key advantage**, only rate reduction.

**Acceptance:** MDI key-rate curve reproduces published behaviour within an order of magnitude; the blinding demonstration is a passing test, not a plot.

### 3.2 WS-2 — Source-side telemetry simulator (the new physics)

New module `src/simulator/source_side.py`. Model Alice's transmitter as a component chain emitting observable telemetry alongside the protocol statistics.

**Telemetry channels:**

| Signal | Units | Primarily reveals |
|---|---|---|
| `watchdog_power` | nW | Trojan-horse back-reflection |
| `isolation_db` | dB | Laser damage (irreversible degradation) |
| `smsr_db`, `rin`, `wavelength_shift_pm` | dB, —, pm | Laser seeding / injection locking |
| `modulator_bias_v`, `extinction_ratio_db` | V, dB | Induced photorefraction |
| `qber_z`, `qber_x`, `gain`, decoy yields | — | Protocol-level consequences of all of the above |

**Attack classes:** 0 Normal · 1 Trojan-horse · 2 Laser seeding · 3 Induced photorefraction · 4 Laser damage · 5 Laser-damage → THA **chain**.

**Physics to get right:** THA leakage bounded by outgoing photon number $\mu_{\text{out}}$ (Lucamarini et al., *PRX* 2015), with back-reflected power $P_{\text{wd}} = P_{\text{in}} R \cdot 10^{-I/10}$; photorefraction as a saturating exponential with time constant τ and slow dark recovery; laser damage as a monotonic non-recovering isolation drop that raises the success probability of the subsequent THA.

**Acceptance:** each attack's effect on $\mu_{\text{out}}$ and on QBER is derived from the component model, not hard-coded; benign nuisance drift (thermal, aging) overlaps the attack ranges so the classes are not trivially separable.

### 3.3 WS-3 — Multi-modal temporal ML + XAI

Generalize `CausalMultiAttackFeatureExtractor` from a single QBER stream to **multi-channel telemetry**, applying the existing 63-feature bank per channel and adding cross-channel features (notably watchdog × modulation-clock correlation for THA).

**Models:** XGBoost and Random Forest. **Baselines:** fixed threshold per telemetry channel, CUSUM changepoint detection, and the Hoeffding test — to show that single-channel thresholds miss the slow-drift attacks.

**Reported:** accuracy, macro-F1, per-class F1, FNR, detection latency ΔT, FPR calibrated at 1%, SHAP attribution **by telemetry channel**, and leave-one-modality-out ablation.

**Acceptance:** SHAP identifies the physically correct channel for each attack (e.g. `modulator_bias_v` drift for photorefraction, `isolation_db` for laser damage); latency for the slow-drift attacks is reported in physical time, not just blocks.

### 3.4 WS-4 — Integration and deliverables

Coverage matrix (Section 2.2) as a publication figure showing which layer stops which attack; extended SOC dashboard with a source-side telemetry panel; `docs/PHYSICS.md` and `docs/THREAT_MODEL.md` in plain language; consolidated limitations.

### 3.5 Sequencing

WS-1 and WS-2 are independent and can proceed in parallel. WS-3 depends on WS-2. P2.0 close-out should precede everything, because the leakage fix changes every number Phase 2 reports.

---

## 4. Repository Architecture

```text
qkd-explainable-anomaly-detection/
├── .gitignore
├── README.md
├── requirements.txt
├── config.yaml                  # Simulation & ML hyperparameters
├── app.py                       # Interactive Streamlit SOC dashboard (6 tabs, scenario-driven)
├── data/
│   ├── raw/                     # Raw simulated session sequences (.parquet / .csv)
│   └── processed/               # Sliding-window time-series dataset (.csv)
│       └── trial_sessions/      # Replayable one-per-class sessions used by the dashboard
├── results/
│   ├── figures/                 # High-res publication plots (Latency, ROC, SHAP, Generalization)
│   └── metrics/                 # JSON/CSV benchmark tables for the IEEE paper
├── src/
│   ├── simulator/
│   │   ├── bb84_engine.py       # Alice state prep, Bob measurement, Sifting, QBER calculation
│   │   ├── noise_models.py      # Fiber (Stationary Depolarizing) & Free-Space (Log-Normal Scintillation)
│   │   ├── eve_attacks.py       # 5 classes: Benign, Fractional IR, PNS, Fake State, Time-Shift
│   │   ├── tf_qkd.py            # SNS Twin-Field QKD, decoy-state BB84 key rate, PLOB bound
│   │   ├── mdi_qkd.py           # MDI-QKD relay model + detector-attack comparison
│   │   └── source_side.py       # [Phase 3] Transmitter telemetry & light-injection attacks
│   ├── features/
│   │   └── time_series.py       # Causal sliding-window extraction (14-feature and 63-feature banks)
│   ├── models/
│   │   ├── hoeffding_baseline.py# Lee et al. (2022) classical dynamic threshold
│   │   ├── ml_classifiers.py    # Random Forest & XGBoost wrappers with calibrated FPR
│   │   └── explainability.py    # SHAP TreeExplainer integration & feature attribution
│   └── evaluation/
│       └── benchmarks.py        # Latency (Delta T), FPR control, Cross-channel degradation
├── scripts/
│   ├── 01_verify_physics.py     # Analytical and Qiskit sanity check for BB84 physics
│   ├── 02_generate_dataset.py   # Multi-condition synthetic dataset generator
│   ├── 03_run_benchmarks.py     # Train, evaluate latency & cross-channel shift
│   ├── 04_explain_and_plot.py   # Generate SHAP plots and 300 DPI publication figures
│   ├── 05_modern_qkd_tfqkd.py   # Twin-Field QKD: key rate vs distance, phase-attack detection
│   ├── 06_generate_multiattack_dataset.py  # Phase 2: 5-class multi-attack dataset generator
│   ├── 07_train_multiattack_classifiers.py # Phase 2: Train & tune multi-class XGBoost
│   ├── 08_evaluate_multiattack.py          # Phase 2: Benchmark metrics, confusion matrix & ROC
│   ├── 09_explain_and_ablation.py          # Phase 2: SHAP ranking & feature group ablation
│   ├── 10_multiseed_evaluation.py          # P2.1: leakage audit + multi-seed mean/std benchmarks
│   └── 11_mdi_protocol_comparison.py       # MDI-QKD curves, detector attack, trial datasets
└── tests/                       # pytest suite: 32 tests (BB84, TF-QKD, MDI-QKD, 5-class, leakage guard)
```

Entries marked `[Phase 3]` are specified in Section 3 and not yet implemented.

---

## 5. Quickstart Guide

### Prerequisites
- Python 3.10+
- Recommended virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Reproducing Experimental Results

Execute the sequential experimental pipeline:

```bash
# 1. Verify quantum state physics & channel error equations
python scripts/01_verify_physics.py

# 2. Generate multi-regime synthetic time-series datasets
python scripts/02_generate_dataset.py

# 3. Train models and evaluate Latency (Delta T) & Cross-Channel Generalization
python scripts/03_run_benchmarks.py

# 4. Generate SHAP explainability analyses and publication figures
python scripts/04_explain_and_plot.py

# 5. Modern protocol: Sending-or-Not-Sending Twin-Field QKD
python scripts/05_modern_qkd_tfqkd.py

# 6. Phase 2: Generate 5-class multi-attack dataset (63 features)
python scripts/06_generate_multiattack_dataset.py

# 7. Phase 2: Train multi-class XGBoost classifier
python scripts/07_train_multiattack_classifiers.py

# 8. Phase 2: Evaluate benchmarks, confusion matrix & ROC curves
python scripts/08_evaluate_multiattack.py

# 9. Phase 2: SHAP explainability & feature group ablation
python scripts/09_explain_and_ablation.py

# 10. P2.1: leakage audit and multi-seed benchmarks (mean +/- std over 5 seeds)
python scripts/10_multiseed_evaluation.py

# 11. MDI-QKD curves, detector-attack demo, and trial datasets
python scripts/11_mdi_protocol_comparison.py

# 12. Run the interactive Streamlit SOC dashboard
python -m streamlit run app.py

# Run the test suite (32 tests)
python -m pytest
```

Approximate runtimes on a 24-core desktop: 01 ≈ 2 s, 02 ≈ 70 s, 03 ≈ 60 s, 04 ≈ 10 s, 05 ≈ 3 min.

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

## 10. Phase 2: Multi-Attack Classification & Streamlit SOC Dashboard
*Reference Architecture:* *"Beyond the QBER Threshold: A Temporal QBER Based Machine Learning Framework for Multi Attack Detection in BB84 QKD"*

Phase 2 scales the framework from binary anomaly detection to a **5-class multi-attack classification engine** powered by XGBoost and a real-time **Streamlit Security Operations Center (SOC)**.

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

### 10.3 Experimental Benchmarks (leakage-free, 5 independent seeds)

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

### 10.4 Running the Streamlit Security Operations Center (SOC)
```bash
python -m streamlit run app.py
```
Six tabs, scenario-driven:
* **Start Here** - the problem in plain language, the corrected headline numbers, and the protocol decision table.
* **Live Monitor** - six presets, each chosen to expose a different blind spot of the 11% rule (clean baseline, sub-threshold stealth, zero-QBER PNS, detector blinding where the QBER *drops*, time-shift basis split, and an attack on a drifting free-space link). Replays pre-generated trial sessions instantly or simulates live.
* **Why? (SHAP)** - per-block attribution with group-level roll-up and a plain-language reading, including when the block was misclassified.
* **MDI-QKD** - interactive sweep of how much of the relay Eve controls, side by side for BB84 and MDI-QKD, plus the key-rate curves.
* **Benchmarks** - corrected metrics, the leakage story, and the ablation.
* **Threat Atlas** - per-class physics, the Phase 3 source-side attacks, the 63 features, the nuisance ranges, and the limitations.

Trial sessions are generated by `scripts/11_mdi_protocol_comparison.py` into `data/processed/trial_sessions/` and carry the model's per-block predictions, so the dashboard replays them with no simulation cost. They are drawn from the same leakage-free distribution the model was trained on - the dashboard passes `randomize_nuisance=True`, which the earlier version did not, so its live sessions no longer come from a different distribution than the model saw in training.

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

### 10.4 Running the Streamlit Security Operations Center (SOC)
```bash
streamlit run app.py
```
* **Real-Time Temporal Monitor:** Live streaming QBER chart with the 11% threshold line and dual-basis ($Q_Z$ vs $Q_X$) error tracker.
* **XGBoost Threat HUD:** Live 5-class confidence bars, detection latency counter ($\Delta T$), and automated key-abort simulation.
* **SHAP Explainability Card:** On-the-fly waterfall plot showing exact physics feature contributions.
* **Paper Visualizer:** Embedded 300 DPI confusion matrix, ROC curves, and ablation tables.

---

## 11. Mathematical Formulations

### Hoeffding Baseline Threshold (Lee et al., 2022)
For a sifted key block of size $N$ with baseline error $e_0$ and significance level $\alpha = 0.01$:
$$\hat{e}_{\text{thresh}} = e_0 + \sqrt{\frac{\ln(1/\alpha)}{2N}}$$

### Causal Feature Extraction
Given discrete error sequence $E = [e_1, e_2, \dots, e_t]$, feature vectors $x_t \in \mathbb{R}^d$ are computed strictly using backward sliding windows $E_t = [e_{t-W+1}, \dots, e_t]$ to ensure zero future data leakage. Extracted features include current QBER, rolling mean and standard deviation, Fano factor, first-order drift slope, EMA divergence, rolling skewness, kurtosis, Shannon entropy, burstiness index and lag-1 autocorrelation.

---

## 12. Limitations

* **Everything is simulated.** There is no hardware in this project, and no measurement from a physical QKD system.
* **Phase 2 accuracy has been corrected** from 99.93% to 85.63% ± 1.36% after removing label leakage (Section 10.5). Residual single-feature separability is about 0.93 AUC, reflecting genuine physical differences between the attacks rather than simulator artefacts.
* **Attack models are phenomenological.** The fake-state, time-shift and PNS models reproduce the qualitative signature of each attack rather than deriving it from a full device model. The TF-QKD phase-tampering attack is likewise phenomenological, not an optimal collective attack.
* **Key rates are asymptotic** — no finite-key corrections and no AOPP.
* **Free-space drift** is a log-normal AR(1) process on the QBER, not a full turbulence model.
* **MDI-QKD key rates are asymptotic** - no finite-key correction, infinite-decoy limit, and `Y11`/`e11` are leading-order expressions. The detector-attack comparison in Section 2.6 models Eve's best strategy analytically rather than simulating her hardware.
* **Phase 3 is mostly a specification.** WS-1 (the MDI-QKD model) is implemented; WS-2 to WS-4 - the source-side telemetry simulator and the multi-modal monitor - are not.

---

## 13. Citation

If you use this benchmark codebase in your research, please cite:
```bibtex
@inproceedings{qkd_explainable_temporal_ml_2026,
  title={Explainable ML for Eavesdropping Detection in Quantum-Key Distribution (BB84)},
  author={Kompi, Samarth},
  booktitle={Undergraduate Experiential Learning Conference in Quantum Engineering and Computing},
  year={2026}
}
```
