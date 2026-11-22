# Codebase Audit (2026-11-22)

Written before any of the improvement work in this branch, against commit `edaa8f7`.
Severity: **H** changes a reported number or invalidates a claim, **M** biases results or
reproducibility, **L** cosmetic or fragile.

## 1. What each module does

| Module | Role |
|---|---|
| `src/simulator/bb84_engine.py` | Block-level BB84: random bits/bases, Eve, detection, sifting, per-basis QBER, error-gap stats. `simulate_session` draws per-session nuisance values from `NUISANCE_RANGES` and switches Eve on at `attack_start_block`. Also a Qiskit cross-check. |
| `src/simulator/eve_attacks.py` | Five behaviours in `execute_full_attack`: benign, intercept-resend, PNS, fake-state (blinding), time-shift. |
| `src/simulator/noise_models.py` | `FiberNoiseModel` (constant bit-flip rate), `FreeSpaceNoiseModel` (AR(1) log-normal drift of the *QBER*). |
| `src/simulator/tf_qkd.py`, `mdi_qkd.py` | Asymptotic key-rate models (SNS-TF, MDI, decoy BB84, PLOB) and an analytic detector-attack comparison. |
| `src/features/time_series.py` | Legacy 14-feature extractor (W=10) and the 63-feature multi-attack extractor (W=50), both causal. |
| `src/models/*` | Hoeffding threshold, RF/XGB wrappers with FPR calibration, SHAP wrapper. |
| `src/evaluation/benchmarks.py` | Binary metrics, detection latency, cross-channel matrix (legacy 3-class pipeline). |
| `scripts/01–05` | Legacy pipeline: physics checks, 3-class dataset, benchmarks, figures, TF-QKD. |
| `scripts/06–11` | Multi-attack pipeline: 5-class dataset, XGB training, evaluation, SHAP/ablation, multi-seed, MDI + trial sessions. |
| `backend/`, `frontend/` | FastAPI + no-build dashboard. |

## 2. Where each CSV comes from

| File | Script | Model / split |
|---|---|---|
| `classification_benchmarks_fiber.csv` | `03_run_benchmarks.py` | Legacy 3-class, binary attack-vs-benign, session split |
| `cross_channel_generalization.csv` | `03_run_benchmarks.py` | Same models, fiber → free-space |
| `detection_latency_comparison.csv` | `03_run_benchmarks.py` | Fresh sessions per p_eve, 40 per point |
| `feature_importance_shap.csv` | `04_explain_and_plot.py` | Legacy RF binary model |
| `multiattack_feature_importance_shap.csv`, `multiattack_feature_ablation.csv` | `09_explain_and_ablation.py` | 5-class XGB, single seed, ablation on a 50k-row subsample |
| `multiattack_multiseed_per_seed.csv`, `_summary.csv`, `multiattack_leakage_audit.json` | `10_multiseed_evaluation.py` | 5 seeds × 400 sessions/class, XGB only |
| `multiattack_overall_benchmarks.csv`, `multiattack_per_class_f1.csv` | `08_evaluate_multiattack.py` | Single seed (42) |
| `cv_multiattack_xgboost.json` | `07_train_multiattack_classifiers.py` | Row-level 5-fold CV (see L-2) |
| `mdi_key_rate_vs_distance.csv`, `mdi_detector_attack_comparison.csv` | `11_mdi_protocol_comparison.py` | Analytic, no sampling |
| `tfqkd_*.csv` | `05_modern_qkd_tfqkd.py` | Analytic + finite-block sampling |

## 3. Bugs

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| B-1 | **H** | **Dark counts never cause bit errors.** A dark click on an undetected pulse is assigned Alice's correct bit whenever the bases match. | Block with T = 1e-6, p_dark = 0.05: 4,909 sifted bits, QBER **0.000** (should be ≈ 0.5). Dark counts therefore cannot raise QBER at long distance, and the model never sees that regime. |
| B-2 | **H** | **Inconsistent photon-number models between classes.** Benign, IR, fake-state and time-shift treat every pulse as exactly one photon (detect with p = T·η). Only the PNS class samples Poisson photon numbers, including vacuum. | T = 0.5: benign detects 49.9% of pulses; a weak coherent pulse (WCP) with μ = 0.5 would detect 22.1%; PNS detects 7.6%. The PNS class is separable by the *missing vacuum pulses in the other classes*, not by Eve. |
| B-3 | M | Detection probability is linear (T·η) instead of 1 − exp(−μTη); multi-photon clicks and double clicks are absent. | Code: `rng.random() < base_eff`. |
| B-4 | M | `EveAttacker(class_id=2)` silently becomes intercept-resend when `p_eve ∈ {0.2, 1.0}` with default kwargs (a test-compatibility hack). | `eve_attacks.py` constructor. |
| B-5 | L | `dark_count_ratio = 1e-5 / qber` uses a constant, not the simulated dark-count rate, so it is just 1/QBER. | `time_series.py`. |
| B-6 | L | `FreeSpaceNoiseModel` drifts the *error rate*; transmittance stays constant, so turbulence never causes fading or loss of counts. | `noise_models.py`. |

## 4. Leakage and evaluation risks

| ID | Sev | Finding |
|---|---|---|
| L-1 | **H** | **Benign sessions are perfectly stationary.** Transmittance, base QBER and detector efficiencies are fixed for the whole session, while every attack switches on as a step at block 76. Any change detector separates the classes. The top SHAP feature is `loss_gradient` (1.29, 3× the next feature): the model largely learned "something changed at onset", which a real channel also does (polarisation drift, connector reseat, temperature). |
| L-2 | **H** | `07` cross-validates with `StratifiedKFold` over **rows**. Overlapping W = 50 windows from one session land in both folds. CV accuracy is 90.6% while the session-held-out test is 86.3%. Train/test files themselves are session-split, which is correct. |
| L-3 | M | `08` puts the fixed-11% baseline (binary task, accuracy 0.41) and XGBoost (5-class task, accuracy 0.86) in one table, so the two "accuracy" columns measure different tasks. Its FNR column is comparable; its accuracy column is not. |
| L-4 | M | Single-feature AUC audit is computed only on the test split and only for attack classes, and its maximum (0.93) sits just under the 0.95 alarm level. The feature that hits it is not reported in the summary CSV. |
| L-5 | M | Window labels use the class of the *last* block, so the first ~49 attacked windows mix pre-attack data. This lowers accuracy and inflates latency rather than leaking, but it is undocumented. |
| L-6 | M | Only XGBoost is evaluated multi-seed; there is no SVM or RF on the 5-class task, no confidence intervals, and no calibration analysis. |
| L-7 | L | No sidecar metadata (seed, config hash, git commit) for any dataset; `data/processed` CSVs cannot be tied to the code that made them. |

## 5. Unrealistic or unsourced parameters

| Parameter | Current | Realistic (to be cited in `config/physics.yaml`) |
|---|---|---|
| Channel transmittance | 0.10–0.95, no distance | 10^(−αL/10), α ≈ 0.2 dB/km, L ≈ 0–100 km → 1.0–0.01 |
| Detector efficiency | 0.80–1.00 | InGaAs SPAD ≈ 0.10–0.30; SNSPD ≈ 0.60–0.95 |
| Dark-count probability | 1e-6–5e-5 per gate | InGaAs ≈ 1e-6–1e-5 per gate; SNSPD ≈ 1e-8–1e-7 per gate |
| Intrinsic error e_d | base QBER 0.5–6%, unattributed | Optical misalignment ≈ 1–2%; total QBER then follows from e_d, dark counts and loss |
| Pulses per block | 1,024 raw → ~250 sifted bits | Fine for speed, but the per-block QBER then has a 95% CI of roughly ±2.5 percentage points, so per-block threshold decisions are over-trusted (no finite-size treatment anywhere) |
| Mean photon number | μ only used by PNS | All classes should share one WCP source |
| Fake-state dark counts | dark × 0.1 | A class-specific constant; acts as a fingerprint |
| Afterpulsing, dead time | absent | Optional, ≈ 1–5% afterpulse for gated InGaAs |

## 6. Other observations

* `mdi_detector_attack_comparison.csv` is analytic and phenomenological: Eve's information is set equal to her control fraction, and the BB84 side assumes 0% honest error while the MDI side assumes 0.5%. It illustrates the mechanism; it is not a simulation.
* `data/raw/` holds untracked files that look like real QKD telemetry (QBER and secure-key-rate sweeps with VOA attenuation and ROADM-injected noise; summaries name Toshiba LE and IDQ Clavis systems). Their source is not recorded anywhere in the repo. Only 1 of the 10 CSVs the summaries describe is present. See `DATASETS.md`.
* `requirements.txt` uses lower bounds only (`>=`), so results drift with library versions, as already seen in the 4th-decimal ML differences.
* There is no `qcloud.env` entry in `.gitignore`.

## 7. What this branch changes

Sections 1–7 of the improvement plan address B-1, B-2, B-3, B-6, L-1, L-2, L-3, L-4, L-6, L-7 and the parameter table. Each change and its effect on metrics are recorded in `CHANGELOG.md`.
