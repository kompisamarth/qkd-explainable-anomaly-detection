# Explainable ML for Eavesdropping Detection in Quantum Key Distribution (BB84)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE Format](https://img.shields.io/badge/Paper-IEEE%20EL%20Conference-orange.svg)](#)

A research pipeline and benchmark implementation for explainable, low-latency machine learning detection of stealthy eavesdropping in discrete-variable Quantum Key Distribution (BB84).

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
│   │   └── eve_attacks.py       # Benign (p=0), Fractional IR (0 < p < 1), Full IR (p=1)
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
    └── 04_explain_and_plot.py   # Generate SHAP plots and 300 DPI publication figures
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
```

---

## 5. Mathematical Formulations

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

## 6. Citation

If you use this benchmark codebase in your research, please cite:
```bibtex
@inproceedings{qkd_explainable_temporal_ml_2026,
  title={Explainable ML for Eavesdropping Detection in Quantum-Key Distribution (BB84)},
  author={Kompi, Samarth},
  booktitle={Undergraduate Experiential Learning Conference in Quantum Engineering and Computing},
  year={2026}
}
```
