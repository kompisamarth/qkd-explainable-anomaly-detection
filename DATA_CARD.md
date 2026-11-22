# Data Card: Calibrated Multi-Attack BB84 Dataset

**All rows are simulated.** No row comes from QKD hardware. See `LIMITATIONS.md`. Attack models: `docs/ATTACK_MODELS.md`.

## Files (in `data/processed/`, produced by `scripts/06_generate_multiattack_dataset.py`)

| File | Rows are | Purpose |
|---|---|---|
| `train_multiattack.{csv,parquet}` | sliding windows (W = 50 blocks) | Training and GroupKFold cross-validation |
| `test_multiattack.{csv,parquet}` | windows | Test: same distributions, sessions never seen in training |
| `heldout_generalization.{csv,parquet}` | windows | Generalization test: attack intensities and fiber distances outside the training ranges |
| `sessions_multiattack.csv` | one row per session | The physical parameters drawn for each session, and its split |
| `*.meta.json` | sidecar | Seed, config hash, git commit, UTC generation time, row/session/class counts, parameter ranges, feature list |

Train and test are split by **session id**, stratified by class (default 80/20 of sessions);
no session contributes rows to both. Each session has its own seed derived from the dataset
seed with `numpy.random.SeedSequence`, so the data is identical for any number of workers.

## Generation process

One session = 150 blocks of 500,000 weak coherent pulses (0.5 ms at 1 GHz). The channel,
detector and noise are drawn **from the same distribution for every class**
(`config/physics.yaml → calibrated_bb84`). Eve switches on at block 76 (attacked sessions only).
Every session, attacked or not, also has misalignment drift, detector-efficiency drift and,
with probability 0.35, one loss step at a random block, so "the channel changed" is not an
attack signature by itself.

Classes (`block_class` / `session_attack_class`):
0 Normal, 1 intercept-resend, 2 PNS, 3 detector blinding, 4 time-shift,
5 Trojan horse (simplified), 6 beam splitting. See `docs/ATTACK_MODELS.md`.

## Columns

### Identity and labels
| Column | Unit | Meaning |
|---|---|---|
| `session_id` | — | Unique per session; the grouping key for every split |
| `block_index` | block | Index (1-based) of the **last** block in the window |
| `channel_type` | — | `fiber` or `free_space` |
| `session_attack_class` | class id | Attack the session switches to at onset (0 for benign sessions) |
| `session_p_eve` | see attack | The session's attack intensity (fraction intercepted, tapped fraction, …) |
| `attack_start_block` | block | Onset block |
| `is_attack_active` | 0/1 | 1 when the last block is at or after onset in an attacked session |
| `block_class` | class id | **Training label**: `session_attack_class` if the last block is attacked, else 0. Windows that straddle the onset mix pre- and post-attack blocks |

### Metadata that is never a model input
| Column | Unit | Meaning |
|---|---|---|
| `qber_block` | fraction | QBER of the last block alone |
| `sifted_bits_block` | bits | Sifted signal bits in the last block (Z + X) |
| `qber_ci_low`, `qber_ci_high` | fraction | 95% Wilson score interval of `qber_block` |
| `eve_info_fraction` | fraction | Rough estimate of the share of sifted bits Eve learns (ground truth, not observable) |
| `true_transmittance` | fraction | Channel transmittance of the last block (ground truth) |

### Model features (63, window of W = 50 blocks unless stated)
Notation: q_t is the QBER of block t, computed on signal pulses whose bases matched; q_Z,
q_X are the same per basis; η_t is the click probability per signal pulse (clicks ÷ pulses).

| Group | Feature | Unit | Computation |
|---|---|---|---|
| Statistical | `qber_mean`, `qber_variance`, `qber_std` | fraction, fraction² | mean, variance (ddof 1) and std of q_t |
| | `qber_skewness`, `qber_kurtosis` | — | bias-corrected skew / excess kurtosis of q_t (0 if constant) |
| | `qber_iqr` | fraction | 75th − 25th percentile of q_t |
| | `qber_cv` | — | std / mean |
| Burst & instability | `spike_density` | fraction | share of blocks with q_t > mean + 2·std |
| | `burst_duration` | blocks | longest run with q_t > mean + std |
| | `jump_energy` | fraction² | Σ (q_t − q_{t−1})² |
| | `temporal_drift_slope` | fraction / block | least-squares slope of q_t |
| | `max_jump`, `jump_count` | fraction, count | largest abs. jump; jumps > 1.5·std |
| | `error_gap_mean`, `error_gap_std`, `error_gap_cv` | bits, bits, — | window mean of the per-block mean / std / CV of the distance between consecutive errors in the sifted string (first 4,096 sifted bits) |
| | `burstiness_index` | — | (std − mean) / (std + mean), clipped to [−1, 1] |
| Spectral & temporal | `autocorr_lag_1` … `autocorr_lag_5` | — | autocorrelation of q_t at lags 1–5 |
| | `dft_low_energy`, `dft_mid_energy`, `dft_high_energy` | fraction of power | share of DFT power (mean removed) in the lower / middle / upper third of frequencies |
| | `spectral_entropy` | bits | entropy of the normalized DFT power |
| | `spectral_centroid` | bin index | power-weighted mean frequency bin |
| | `shannon_entropy` | bits | entropy of q_t over 5 bins in [0, 0.35] |
| | `ema_divergence` | fraction | q_t of the last block − EMA (span 10) |
| Basis-dependent | `qber_rectilinear`, `qber_diagonal` | fraction | window mean of q_Z, q_X |
| | `basis_asymmetry`, `basis_ratio` | fraction, — | abs(q_Z − q_X); q_Z / q_X |
| | `basis_burst_energy_z`, `basis_burst_energy_x` | fraction² | jump energy per basis |
| | `basis_variance_ratio` | — | var(q_Z) / var(q_X) |
| | `basis_cross_correlation` | — | correlation of q_Z and q_X |
| | `basis_skewness_diff`, `basis_kurtosis_diff` | — | skew(q_Z) − skew(q_X); same for kurtosis |
| | `sift_ratio_z`, `sift_ratio_x` | fraction | share of sifted bits from each basis |
| Channel interaction | `key_generation_rate` | bits / pulse | mean sifted bits ÷ mean pulses per block |
| | `transmission_efficiency` | clicks / pulse | mean η_t |
| | `qber_loss_interaction`, `qber_loss_product` | — | qber_mean / η; qber_mean · (1 − η) |
| | `loss_anomaly_index` | clicks / pulse | η − 0.5 (legacy definition; a constant shift) |
| | `dark_count_ratio` | fraction | expected dark-count clicks (from the detector's specified dark rate) ÷ observed clicks |
| | `multiscale_mean_w{10,25,50}`, `multiscale_std_w…`, `multiscale_fano_w…`, `multiscale_drift_w…` | as above | mean, std, variance/mean and slope of q_t over the last 10 / 25 / 50 blocks |
| | `error_clustering_coefficient` | fraction | window mean of the share of errors that sit next to another error |
| | `loss_gradient` | (clicks / pulse) / block | least-squares slope of η_t |
| | `efficiency_entropy` | bits | entropy of η_t ÷ window mean over 5 bins in [0.5, 1.5] |
| | `qber_stability_index` | — | exp(−abs(qber_mean − 0.03) / 0.03) |

### Optional decoy-state features (only when `source.decoy.enabled: true`)
| Feature | Unit | Computation |
|---|---|---|
| `decoy_gain_ratio` | — | window mean of decoy gain ÷ signal gain |
| `vacuum_gain_ratio` | — | vacuum gain ÷ signal gain |
| `decoy_qber_gap` | fraction | decoy QBER − signal QBER |
| `y1_lower_bound_ratio` | fraction | vacuum + weak-decoy lower bound on the single-photon yield (Ma et al. 2005), as a share of the signal gain |

### `sessions_multiattack.csv`
`session_id, class_id, session_seed, heldout, channel_type, distance_km, loss_db, detector_type,
eta_mean, p_dark, afterpulse, e_d0, mu, scint_index, loss_step_block, loss_step_db, attack,
intensity, atk_*` (attack-specific parameters), `split`.

## Leakage checks

`scripts/06` and `scripts/10` report, for every class, the single feature with the best
one-vs-rest AUC (`results/metrics/multiattack_leakage_audit.json`). AUC > 0.95 is
flagged for review. A flag is not automatically a bug: intercept-resend *should* separate on
QBER, because raising the error rate is what that attack physically does. A flag on a feature
that encodes the simulator rather than physics (a constant, a channel parameter, the onset
time) is leakage.
