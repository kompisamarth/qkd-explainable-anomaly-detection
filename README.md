# Explainable ML for Eavesdropping Detection in Quantum Key Distribution (BB84 + Twin-Field QKD)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE Format](https://img.shields.io/badge/Paper-IEEE%20EL%20Conference-orange.svg)](#)

A research pipeline and benchmark implementation for explainable, low-latency machine learning detection of stealthy eavesdropping in discrete-variable Quantum Key Distribution. It covers the traditional **BB84** protocol and a modern **Sending-or-Not-Sending Twin-Field QKD (SNS-TF-QKD)** simulator, the protocol family behind today's longest-distance fiber QKD records (see [Section 7](#7-modern-protocol-sending-or-not-sending-twin-field-qkd)).

---

## 1. Scientific Overview & Problem Formulation

Standard BB84 QKD networks rely on a **static Quantum Bit Error Rate (QBER) abort threshold** (typically $e_{\text{abort}} \approx 11\%$, derived from the Shor-Preskill security bound) or scalar statistical bounds (such as Hoeffding's inequality) to flag potential eavesdropping.

However, intelligent adversaries (Eve) can execute **fractional/stealthy intercept-resend (IR) attacks**, intercepting only a proportion $p_{\text{eve}} \in [0.10, 0.30]$ of transmitted photon pulses. Because the induced error $\Delta e \approx \frac{1}{4} p_{\text{eve}} (1 - 2 e_0)$ produces a total QBER of $\approx 5.5\% - 9.5\%$, it remains completely beneath the static $11\%$ threshold, leaking partial key information without triggering a hardware abort.

While modern ML classifiers can reach high static classification accuracy, literature frequently overlooks three critical operational imperatives:
1. **Detection Latency ($\Delta T = T_{\text{detect}} - T_{\text{attack}}$)**: How many elapsed sifted photon blocks occur before the detector issues a true alarm?
2. **Cross-Channel Noise Generalization**: How does a model trained under stationary depolarizing fiber noise ($e_0 = 3\%$) degrade when deployed onto non-stationary free-space/satellite links subject to log-normal atmospheric turbulence and scintillation drift?
3. **Transparent Explainability (XAI)**: Can the detector provide verifiable, feature-level attribution (via SHAP TreeExplainer) justifying an autonomous key-abort decision?

---

## 2. Protocol Classes & Threat Model

| Class | Label | Eavesdropping Strategy | Interception Rate $p_{\text{eve}}$ | Expected QBER ($e_0 = 3\%$) |
|---|:---:|---|:---:|:---:|
| **Class 0** | `0` | **Benign Noise** (Stationary fiber or scintillation) | $0.0$ | $\approx 3.0\%$ |
| **Class 1** | `1` | **Full Intercept-Resend (IR)** | $1.0$ | $\approx 26.5\%$ (Exceeds $11\%$ abort) |
| **Class 2** | `2` | **Fractional/Stealthy IR** | $[0.10, 0.30]$ | $\approx 5.3\% - 9.9\%$ (**Sub-threshold stealth**) |

---

## 3. Repository Architecture

```text
qkd-explainable-anomaly-detection/
├── .gitignore
├── README.md
├── requirements.txt
├── config.yaml                  # Simulation & ML hyperparameters
├── data/
│   ├── raw/                     # Raw simulated session sequences (.parquet / .csv)
│   └── processed/               # Sliding-window time-series dataset (.csv)
├── results/
│   ├── figures/                 # High-res publication plots (Latency, ROC, SHAP, Generalization)
│   └── metrics/                 # JSON/CSV benchmark tables for the IEEE paper
├── src/
│   ├── simulator/
│   │   ├── bb84_engine.py       # Alice state prep, Bob measurement, Sifting, QBER calculation
│   │   ├── noise_models.py      # Fiber (Stationary Depolarizing) & Free-Space (Log-Normal Scintillation)
│   │   ├── eve_attacks.py       # Benign (p=0), Fractional IR (0 < p < 1), Full IR (p=1)
│   │   └── tf_qkd.py            # SNS Twin-Field QKD, decoy-state BB84 key rate, PLOB bound
│   ├── features/
│   │   └── time_series.py       # Causal sliding-window feature extraction (no future leakage)
│   ├── models/
│   │   ├── hoeffding_baseline.py# Lee et al. (2022) classical dynamic threshold
│   │   ├── ml_classifiers.py    # Random Forest & XGBoost wrappers with calibrated FPR
│   │   └── explainability.py    # SHAP TreeExplainer integration & feature attribution
│   └── evaluation/
│       └── benchmarks.py        # Latency (Delta T), FPR control, Cross-channel degradation
└── scripts/
    ├── 01_verify_physics.py     # Analytical and Qiskit sanity check for BB84 physics
    ├── 02_generate_dataset.py   # Multi-condition synthetic dataset generator
    ├── 03_run_benchmarks.py     # Train, evaluate latency & cross-channel shift
    ├── 04_explain_and_plot.py   # Generate SHAP plots and 300 DPI publication figures
    ├── 05_modern_qkd_tfqkd.py   # Twin-Field QKD: key rate vs distance, phase-attack detection
    ├── 06_generate_multiattack_dataset.py  # Phase 2: 5-class multi-attack dataset generator
    ├── 07_train_multiattack_classifiers.py # Phase 2: Train & tune multi-class XGBoost
    ├── 08_evaluate_multiattack.py          # Phase 2: Benchmark metrics, confusion matrix & ROC
    └── 09_explain_and_ablation.py          # Phase 2: SHAP ranking & feature group ablation
app.py                           # Phase 2: Interactive Streamlit Security Operations Center (SOC)
tests/                           # pytest suite: 23 tests (BB84, Twin-Field, 5-class attacks)
```

---

## 4. Quickstart Guide

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

# 10. Run the interactive Streamlit SOC dashboard
python -m streamlit run app.py

# Run the test suite (23 tests)
python -m pytest
```

Approximate runtimes on a 24-core desktop: 01 ≈ 2 s, 02 ≈ 70 s, 03 ≈ 60 s, 04 ≈ 10 s, 05 ≈ 3 min.

---

## 5. Results (reproduced 2026-10-26)

All numbers below come from running scripts 01-05 with `random_seed: 42` (Python 3.13, numpy 2.1, scikit-learn 1.9, xgboost 3.4).

### 5.1 Physics verification (`01_verify_physics.py`)

| Check | Empirical | Theory |
|---|:---:|:---:|
| Noiseless, no Eve | QBER 0.0000 | 0 |
| Full IR, $e_0=0$ | 0.2533 | 0.2500 |
| Fractional IR, $e_0=3\%$, $p_{\text{eve}}=0.2$ | 0.0780 | 0.0770 |
| Free-space drift: QBER std / lag-1 autocorrelation | 0.0063 / 0.70 | ≈0.009 / 0.75 |
| Qiskit measure-and-resend, $p_{\text{eve}}=1$ | 0.2446 | 0.2500 |

### 5.2 Fiber test set, binary detection (`03_run_benchmarks.py`)

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

### 5.3 Cross-channel generalization (trained on fiber, tested on free-space)

| Detector | Fiber F1 | Free-space F1 | Free-space FPR |
|---|:---:|:---:|:---:|
| Hoeffding (windowed W=10) | 0.986 | 0.971 | 0.59% |
| Random Forest | 0.999 | 0.901 | **11.6%** |
| XGBoost | 0.999 | 0.931 | **7.7%** |

ML detectors trained on stationary fiber noise mistake atmospheric drift for an attack. Their FPR rises far above the 1% target, while the analytic Hoeffding test degrades gracefully. Multi-class models (`multiclass_benchmarks.csv`) drop from 0.999 to 0.92-0.94 accuracy, and the fractional-IR F1 falls to 0.82 (RF) / 0.86 (XGB).

### 5.4 Detection latency ΔT (mean blocks after onset, 40 sessions per point)

| $p_{\text{eve}}$ | Static 11% | Hoeffding single | Hoeffding windowed | RF | XGBoost |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.05 | never | never | 99.7 (2.5% detected) | 8.9 | 13.0 |
| 0.10 | never | never | 9.5 | 1.7 | 1.4 |
| 0.20 | 78.6 (40% detected) | 23.7 | 4.1 | 0.08 | 0.08 |
| 0.30 | 2.3 | 0.7 | 2.5 | 0.0 | 0.0 |

---

## 6. Comparison with the originally committed results

The scripts were first re-run unchanged (apart from a crash fix) and compared with the committed `results/metrics`:

* **Simulation and all threshold baselines reproduced bit for bit.** Static 11% and both Hoeffding variants gave identical confusion matrices and latencies.
* **ML models differed in the 4th decimal.** For example, Random Forest false positives went from 17 to 20 out of 74,600 benign blocks, and latency at $p_{\text{eve}}=0.05$ went from 9.65 to 8.90 blocks (RF) and from 14.03 to 13.00 (XGBoost). This comes from library-version differences (xgboost 3.4, scikit-learn 1.9).
* **SHAP ranking unchanged** (rolling_mean > qber_current > shannon_entropy > rolling_std > burstiness_index); magnitudes within 0.004.

Issues found and fixed during the review:

| # | Issue | Effect | Fix |
|---|---|---|---|
| 1 | `ml_classifiers.py` used `Path` without importing it | Scripts 03 and 04 crashed with `NameError`; the committed results could not be regenerated | Added the import |
| 2 | The Qiskit check never measured Eve: her H gate just stacked with Bob's | The check landed on ~25% by coincidence, not physics | Real measure, collapse and re-prepare in two circuits, plus a zero-QBER no-Eve check |
| 3 | `scintillation_std: 0.008` gave a free-space QBER std of 0.0002, far below the 0.0075 binomial noise of a block | The cross-channel experiment was vacuous (F1 drop ≈ 0.0003), contradicting the claim above | Raised to 0.30 (QBER 3% ± 0.6-0.9%); the degradation in §5.3 is now real |
| 4 | Physics Test 3 seeded the noise model and Eve with the same seed | Noise flips and interceptions were correlated (identical RNG streams) | Distinct seeds. The dataset pipeline was not affected |
| 5 | Each latency detector consumed a different part of the simulator's random stream | Detectors were compared on different sessions | A fresh, identically seeded simulator for each detector |
| 6 | Python loop over 530k windows | Dataset generation took 9.5 min | Vectorized extractor, matching the reference to within 2e-19: **≈ 70 s** |
| 7 | Multi-class models were trained but never evaluated | — | Added `multiclass_benchmarks.csv` and the stealth-subset table |

---

## 7. Modern Protocol: Sending-or-Not-Sending Twin-Field QKD

BB84 key rates fall linearly with channel transmittance η. No point-to-point protocol can beat the **PLOB bound** $-\log_2(1-\eta)$. **Twin-Field QKD** (Lucamarini et al., *Nature* 2018) interferes weak pulses from Alice and Bob at an untrusted relay (Charlie). Only one photon has to survive *half* of the channel, so the rate scales as $\sqrt{\eta}$. The **sending-or-not-sending** variant (Wang, Yu & Hu, *PRA* 2018) underlies record demonstrations such as ~1000 km of fiber (Liu et al., *PRL* 2023). Because Charlie is untrusted, the protocol is also measurement-device-independent, which removes detector side channels.

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

### Limitations

Everything is simulated: there is no hardware, and key rates are asymptotic (no finite-key corrections, no AOPP). The phase-tampering attack is phenomenological, not an optimal collective attack. Free-space drift is modelled as a log-normal AR(1) process on the QBER, not a full turbulence model.

---

## 8. Phase 2: Multi-Attack Classification & Streamlit SOC Dashboard
*Reference Architecture:* *"Beyond the QBER Threshold: A Temporal QBER Based Machine Learning Framework for Multi Attack Detection in BB84 QKD"* (IIT Mandi & DRDO, arXiv:2608.04047v1)

Phase 2 scales the framework from binary anomaly detection to a comprehensive **5-class multi-attack classification engine** powered by a dedicated champion classifier (**XGBoost**) and a real-time **Streamlit Security Operations Center (SOC)**.

### 8.1 The 5 Operational Regimes

| Class | Threat Archetype | Physical Layer Exploited | Primary Feature Fingerprint |
|:---:|---|---|---|
| **0** | **Normal Channel** | Baseline fiber/atmospheric link | Stationary QBER ($\approx 3\%$), symmetric basis errors ($Q_Z \approx Q_X$). |
| **1** | **Fractional IR** | Quantum state vector disturbance | Elevated **jump energy, temporal variance, Fano factor** ($QBER \approx 5.5\% - 9.5\%$). |
| **2** | **PNS Attack** | Laser multi-photon pulse splitting | **Zero added QBER ($\Delta QBER \approx 0$)**, abnormal drop in key generation rate and transmission efficiency. |
| **3** | **Fake State** | SPAD detector blinding into linear mode | **Near-zero QBER ($\approx 0.3\%$)**, deterministic basis-coincidence, abnormal error-gap CV. |
| **4** | **Time-Shift** | Timing & efficiency detector mismatch | Severe **basis asymmetry ($|Q_{\text{rect}} - Q_{\text{diag}}| \gg 0$)** beneath 11% average threshold. |

### 8.2 63 Physics-Informed Feature Categories ($W = 50$)
Extracted over causal backward-looking sliding windows ($W=50$) with zero future leakage:
1. **Statistical (7):** `qber_mean`, `qber_variance`, `qber_std`, `qber_skewness`, `qber_kurtosis`, `qber_iqr`, `qber_cv`.
2. **Burst & Instability (10):** `spike_density`, `burst_duration`, `jump_energy`, `temporal_drift_slope`, `max_jump`, `jump_count`, `error_gap_mean`, `error_gap_std`, `error_gap_cv`, `burstiness_index`.
3. **Spectral & Temporal (12):** `autocorr_lag_1` to `lag_5`, `dft_low_energy`, `dft_mid_energy`, `dft_high_energy`, `spectral_entropy`, `spectral_centroid`, `shannon_entropy`, `ema_divergence`.
4. **Basis-Dependent (12):** `qber_rectilinear` ($q_Z$), `qber_diagonal` ($q_X$), `basis_asymmetry` ($|q_Z - q_X|$), `basis_ratio`, `basis_burst_energy_z`, `basis_burst_energy_x`, `basis_variance_ratio`, `basis_cross_correlation`, `basis_skewness_diff`, `basis_kurtosis_diff`, `sift_ratio_z`, `sift_ratio_x`.
5. **Channel Interaction & Multi-Scale (22):** `key_generation_rate`, `transmission_efficiency`, `qber_loss_interaction`, `qber_loss_product`, `loss_anomaly_index`, `dark_count_ratio`, nested multi-scale statistics (`W=10, 25, 50` mean, std, fano, drift), `error_clustering_coefficient`, `loss_gradient`, `efficiency_entropy`, `qber_stability_index`.

### 8.3 Experimental Benchmarks (101,000 Test Blocks)

| Detector | Accuracy | Macro-F1 | Precision | False Negative Rate (FNR) |
|---|:---:|:---:|:---:|:---:|
| **Conventional Fixed 11% QBER Threshold** | 40.59% | 0.2887 | 0.2030 | **100.0%** (Missed all stealthy attacks) |
| **Proposed XGBoost Multi-Attack Framework** | **99.93%** | **0.9994** | **0.9995** | **0.075%** (FNR reduced by 99.9%) |

#### Per-Class Performance Breakdown (`multiattack_per_class_f1.csv`)
* **Normal Channel:** Precision: 99.89% | Recall: 99.96% | F1: **0.9992**
* **Fractional IR:** Precision: 99.88% | Recall: 99.70% | F1: **0.9979**
* **PNS Attack:** Precision: 99.99% | Recall: 99.99% | F1: **0.9999**
* **Fake State:** Precision: 99.99% | Recall: 99.99% | F1: **0.9999**
* **Time-Shift:** Precision: 100.00% | Recall: 99.99% | F1: **1.0000**

#### Leave-One-Group-Out Feature Ablation Study (`fig_ablation_5class.png`)
* Removing **Channel Interaction** features caused the largest drop (**-0.89%**), followed by **Basis-Dependent** and **Burst/Spectral** features, confirming the DRDO paper's finding that multi-attack discrimination requires channel and basis interaction metrics beyond scalar QBER.

### 8.4 Running the Streamlit Security Operations Center (SOC)
Launch the interactive dashboard with:
```bash
streamlit run app.py
```
* **Real-Time Temporal Monitor:** Live streaming QBER chart with the 11% threshold line and dual-basis ($Q_Z$ vs $Q_X$) error tracker.
* **XGBoost Threat HUD:** Live 5-class horizontal confidence bars, detection latency counter ($\Delta T$), and automated key-abort simulation.
* **SHAP Explainability Card:** On-the-fly waterfall plot showing exact physics feature contributions.
* **Paper Visualizer:** Embedded 300 DPI Confusion Matrix, ROC curves, and ablation tables.

---

## 9. Mathematical Formulations

### Hoeffding Baseline Threshold (Lee et al., 2022)
For a sifted key block of size $N$ with baseline error $e_0$ and significance level $\alpha = 0.01$:
$$\hat{e}_{\text{thresh}} = e_0 + \sqrt{\frac{\ln(1/\alpha)}{2N}}$$

### Causal Feature Extraction
Given discrete error sequence $E = [e_1, e_2, \dots, e_t]$, feature vectors $x_t \in \mathbb{R}^d$ are computed strictly using backward sliding windows $E_t = [e_{t-W+1}, \dots, e_t]$ of length $W=10$ to ensure zero future data leakage. Extracted features include:
- Current QBER ($e_t$)
- Rolling Mean ($\mu_t$) and Rolling Standard Deviation ($\sigma_t$)
- Variance-to-Mean Ratio (Fano factor $\sigma_t^2 / \mu_t$)
- First-Order Linear Drift / Slope across window $W$
- Exponential Moving Average (EMA) divergence ($e_t - \text{EMA}_t$)
- Rolling Skewness, Kurtosis, Shannon Entropy of the error distribution
- Burstiness Index and Lag-1 Autocorrelation

---

## 10. Citation

If you use this benchmark codebase in your research, please cite:
```bibtex
@inproceedings{qkd_explainable_temporal_ml_2026,
  title={Explainable ML for Eavesdropping Detection in Quantum-Key Distribution (BB84)},
  author={Kompi, Samarth},
  booktitle={Undergraduate Experiential Learning Conference in Quantum Engineering and Computing},
  year={2026}
}
```
