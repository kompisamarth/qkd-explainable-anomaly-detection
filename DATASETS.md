# Public QKD Datasets: Survey and Use

Searched 2026-11-22 on Zenodo (REST API, dataset type), IEEE DataPort (site search),
Kaggle (public API) and Figshare. Figshare's search API could not be queried from the
browser used for this survey (CORS), so Figshare is **not covered**; Zenodo and IEEE DataPort
were the productive sources. Only records whose landing page was opened are listed; nothing
here is inferred from a title alone.

## Bottom line

* **No public dataset contains labelled attacks on real QKD hardware.** Every
  attack-labelled QKD dataset found is simulated or synthetic.
* Several datasets contain **real QBER / secure-key-rate telemetry from commercial QKD systems
  in benign operation**. They cannot test attack detection. They can test (a) whether the
  simulator's benign QBER levels are realistic and (b) how often a model trained on
  simulation raises false alarms on real benign telemetry.
* This repository uses Zenodo 21792844 for both, reported separately from simulated results.

## Real measured data

| Dataset | Link | Licence | Protocol / system | Fields | Use here |
|---|---|---|---|---|---|
| QKD Network Characterisation and Coexistence Datasets with Standardised Performance Metrics and Experimental Methods (Horgan, Quigley, Hajar, Briantcev, Kaszubowska-Anandarajah, Kilper; TCD / CONNECT testbed; EU DIGITAL Europe grant 101091520), 2026 | [10.5281/zenodo.21792844](https://doi.org/10.5281/zenodo.21792844) | CC-BY-4.0 | Five commercial systems: Toshiba MU, Toshiba LE (production and research variants), IDQ Clavis3, IDQ ClavisXGR | Per-record QBER and secure key rate (bps); VOA attenuation (dB); fibre length (0 / 50 km); classical noise injection via ROADM (3–12 dBm); perturbation (attenuation-transient) runs. 39 files, 2.2 MB | **Loader + OOD benign test + calibration** (`src/data/real_telemetry.py`). One CSV (`baseline_no_signal_50km_added_fibre_full_sweep.csv`) and three summary JSONs were already in `data/raw/`; the other files are not downloaded. In the local CSV, rows within each VOA setting are **sorted by QBER** and carry no timestamp, so only order-independent statistics can be computed |
| Long-Term Performance Dataset of a Commercial QKD System Operating Under Tropical Climate Conditions in Rio de Janeiro, Brazil (Hermes Quantum Network, IME) | [10.21227/rsgr-z776](https://doi.org/10.21227/rsgr-z776) | Creative Commons Attribution (per landing page) | IDQ Clavis XGR over a 40.3 km spool and a 3.5 km deployed underground link | Timestamped secret key rate, QBER, visibility, detection counts, internal temperatures, external temperature | Best candidate for **time-series** OOD tests (it is timestamped). Not downloaded; IEEE DataPort downloads require an account |
| Underlying data for "Configuration-Dependent Limits of C-Band Quantum–Classical Coexistence over Deployed Metropolitan Fiber" | [10.21227/cmg6-e007](https://doi.org/10.21227/cmg6-e007) | Creative Commons Attribution (per landing page) | Commercial decoy-state BB84 over a 15.5 km field-deployed link with DWDM traffic | QBER, secret key rate, visibility vs. classical perturbation frequency and power | Calibration of coexistence noise; not downloaded |
| Performance Assessment of Fiber-Optic Railway Infrastructure QKD Testbed by Upper Bounded Deviation Method (Weiss) | [10.5281/zenodo.15873852](https://doi.org/10.5281/zenodo.15873852) | CC-BY-4.0 | Two commercial QKD devices (COW and another key protocol) on a railway fibre testbed | `data_COW.txt`, `data_T12.txt` (contents not inspected) | Candidate; not downloaded |
| Replication data for fast quantum key distribution (Ruiz, Velasco) | [10.34810/data190](https://doi.org/10.34810/data190) (Zenodo 14947231) | CC-BY-4.0 | Polarisation-based quantum channel | Photon counts for several channel configurations | Candidate; not downloaded |

## Simulated or synthetic (not usable as real data)

| Dataset | Link | Why excluded |
|---|---|---|
| QF-LOG: Quantum Forensic Dataset for QKD Networks | [10.21227/s0qw-mb45](https://doi.org/10.21227/s0qw-mb45) | Ships the scripts that generate its 50,000 rows; four attack/fault classes are synthetic |
| QKD_Attack_Dataset (Kaggle, MIT) | [kaggle.com/datasets/sattwiksarkar999/qkd-attack-dataset](https://www.kaggle.com/datasets/sattwiksarkar999/qkd-attack-dataset) | Description: "synthetically generated data … based on the BB84 protocol" |
| QKD Smart-Grid Side-Channel Attack Dataset (Kaggle, CC BY-SA 4.0) | [kaggle.com/datasets/mahmoodaljamal/qkd-smart-grid-side-channel-attack-dataset](https://www.kaggle.com/datasets/mahmoodaljamal/qkd-smart-grid-side-channel-attack-dataset) | "generated using a dedicated simulation framework" |
| Q-SEE Cyber-Quantum IoT Dataset (Kaggle, CC BY 4.0) | [kaggle.com/datasets/mahmoodaljamal/q-see-cyber-quantum-iot-dataset](https://www.kaggle.com/datasets/mahmoodaljamal/q-see-cyber-quantum-iot-dataset) | "simulation-generated" |
| DV-QKD in coexistence with classical channels in multicore fiber | [10.5281/zenodo.7015095](https://doi.org/10.5281/zenodo.7015095) | Simulation results (VPIphotonics) |
| Satellite-to-Ground QKD SKR dataset | [10.5281/zenodo.14054879](https://doi.org/10.5281/zenodo.14054879) | Calculated KPIs from a link model |

## How the real data is used (and not used)

1. **Calibration check** (`results/metrics/real_vs_sim_qber_calibration.csv`,
   `results/figures/fig_real_vs_sim_qber.png`).
   * **Real data:** the Toshiba MU coexistence-testbed runs **without** injected classical
     noise give 3.3% QBER at 10 dB total loss, rising to 7.3% at 30 dB on 0 km of added fibre.
     With 50 km added, they give 3.7% at 10 dB and 7.7% at 22 dB.
   * **Simulator at the same loss:** InGaAs SPAD sessions give 2.2% at 10 dB and 6.0% at 30 dB;
     SNSPD sessions stay at 1.6% throughout, because their dark counts are negligible.
   * **Reading:** the real curve has the shape of a dark-count-limited detector, but it sits
     about 1–3 points higher. That implies an intrinsic error near 3% for this system, above the
     1–2% literature range used in `config/physics.yaml`.
   * **Not tuned away.** The gap is reported, not removed: these systems include effects not
     modelled here (afterpulsing at high count rates, their own QBER estimators, residual Raman
     noise). The characterisation summaries for the other systems report session-mean QBERs of
     2.4% (Toshiba LE production), 3.2% (LE research), 2.7% (IDQ Clavis3) and 1.0% (IDQ ClavisXGR).
2. **Out-of-distribution false-alarm test** (`results/metrics/real_data_ood_false_alarms.csv`,
   `real_data_ood_summary.json`, labelled **"Real data (benign, Zenodo 21792844)"**, never mixed
   into simulated metrics).
   * **Setup:** an XGBoost model is trained on the simulated training split using only the
     order-independent QBER statistics (the features the real CSV supports), then applied to
     each real VOA block. Every real block is benign, so every attack prediction is a false alarm.
   * **Result: all 8 local blocks are flagged as intercept-resend (false-alarm rate 100%)**,
     while the fixed 11% threshold raises none. Because the real system's baseline QBER sits
     where the simulator only produces intercept-resend, a model trained on simulation does not
     transfer to this hardware.
   * **Needed before deployment:** recalibrating the benign error distribution to the target
     system, or training on the system's own benign telemetry.
3. **Not used for training.** No real row enters any training set.

## Getting the rest of Zenodo 21792844

```bash
python scripts/12_real_data_evaluation.py --download
```
fetches the 39 files (≈ 2.2 MB, CC-BY-4.0) from the Zenodo API into `data/raw/zenodo_21792844/`.
Cite the dataset as: Horgan J. et al., *QKD Network Characterisation and Coexistence Datasets
with Standardised Performance Metrics and Experimental Methods*, Zenodo (2026),
doi:10.5281/zenodo.21792844.
