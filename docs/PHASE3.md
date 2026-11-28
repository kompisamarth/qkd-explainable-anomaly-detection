# Phase 3: source telemetry and event evaluation

This is a synthetic device-assumption monitor, not a QKD security proof or a hardware-validated detector. Numerical transfer functions, noise, component dynamics and attack powers in `config/source_side.yaml` are explicit assumptions. Papers motivate mechanisms, not the numerical calibration: [Trojan-horse bounds](https://arxiv.org/abs/1506.01989), [laser seeding](https://arxiv.org/abs/1902.09792), [photorefraction](https://doi.org/10.1103/PhysRevApplied.19.054052).

## Reproduction

Use Python 3.13.13 and `requirements-phase3-lock.txt` (NumPy 2.1.3, consistent with the repository bounds). From the repository root:

```powershell
.venv-phase3/Scripts/python -m pytest -q
.venv-phase3/Scripts/python reproduce.py --phase3 --quick
.venv-phase3/Scripts/python scripts/13_source_side_pipeline.py --bounded --seeds 42 43 44 --workers 2 --activate-local
.venv-phase3/Scripts/python scripts/14_source_monitor_replay.py results/phase3/runs/<completed-run>
.venv-phase3/Scripts/python run_dashboard.py --no-browser
```

Omit `--bounded` for the larger five-seed experiment. `--quick` is an execution check, not a benchmark. `--activate-local` selects a completed non-smoke run for the local dashboard; it does not publish anything externally. Reproduction refuses to complete if monitored source files change while running. Failed directories have no completed manifest and cannot be deployed. Historical `full` and `smoke` directories are retained but superseded.

## Measurement and component contract

Minimal sensors: incident watchdog tap, emitted intensity, temperature, modulator bias. Instrumented sensors add isolation, extinction, SMSR, RIN and wavelength diagnostics. Hidden truth includes internal/returned power and simultaneous damage, photorefraction and seeding effects. It is never a model input.

At each integration step: apply incident schedule and any repair; compute wavelength-dependent isolation from the preceding damage state; propagate internal power; update damage, photorefraction and seeding; update thermal/controller state; compute emitted intensity and receiver gain/error; filter sensor states; sample due sensors; retain readings and their saturation status. Thus newly accumulated damage changes internal propagation at the next step. Sensors start settled at an unexposed nominal device, including for startup attacks. Damage persists until explicit repair; other states recover with their configured time constants.

Each sensor has a SHA-256-derived identity RNG, independent of profile/order. Matched devices and scenarios share nuisance histories. Observation age grows while values are held. Stale means no reading or age at least the configured number of sensor reporting periods; missing denotes a missed update, not every subsequent held sample. Saturation remains set until a fresh in-range sample. A change in simulator cadence changes physical integration and sampled receiver pulse count. Receiver pulse rate is an *effective sampling rate*, not a commercial optical clock. Zero detections produce unavailable QBER.

Family labels prioritize the requested active mechanism and otherwise the dominant continuing effect (damage, then photo, then seeding). Separate effect flags prevent interpreting family labels as exclusive physical truth. Chains use configurable second-stage power and wavelength; both stages may have cross-effects. The monitor predicts five families, not a future chain label. The watchdog is an incident-light tap; it has no modulation-clock resolution. No returned-light leakage-to-secret-key bound is computed.

## Features, calibration and metrics

Feature inputs use an allowlist. Regular finite timestamps must match configured cadence; slopes/windows use seconds. No future fill is allowed. The instantaneous ablation uses only current values and algebraic temperature residuals, omitting age, quality/history and rolling terms. The no-environment ablation removes temperature and its residuals, but does not remove physical temperature effects from other measurements.

Training devices fit imputation, scaling and XGBoost. Isolation Forest fits only explicit benign training sessions. Alternating validation device IDs calibrate temperature scaling, confidence rejection and benign score normalization; the remaining validation devices select operating points. Fusion takes the maximum of classifier/anomaly excursions in median-to-benign-95th-percentile units. This preserves the independent anomaly branch and orders values beyond observed calibration maxima. It is not a probability or an exact FAR guarantee. Classifier SHAP explains the uncalibrated class margin, not the anomaly model or the fused score.

All seven detectors (limits, EWMA, CUSUM, anomaly, classifier, fusion, receiver-only) use one threshold grid and one event evaluator. Maximize validation exposure detection subject to the observed pure-benign FAR budget, breaking ties toward lower thresholds. Report actual selected FAR and benign exposure. Receiver normalization uses benign training data and its threshold uses the same validation procedure. Temperature calibration reports ECE, multiclass Brier and log loss on held-out samples. Unknown rejection combines anomaly alarms and benign/low-confidence classification; it is an operational rejection, not proof of an unseen mechanism.

Three consecutive high samples start an alarm. Five low samples at or below 80% of the threshold clear it; intervening exceedances are merged into one alarm episode. The full alarm sequence is computed before scoring. Ordinary sessions score from 60 seconds; an attack already present at time zero scores from zero, without resetting detector history. Warm-up never creates a rising edge. A pre-existing alarm does not earn a new detection at onset, warm-up or the second chain stage. An exposure hit requires a new edge while incident power is nonzero. Pulses within one intermittent schedule count as one episode. Detection after cessation due to continuing component effects is separate. Missed latency is null. Left-censored onsets and right-censored exposure/effects are marked. Startup policy uses truth only for retrospective scoring, never to control the monitor.

Reports include pure-benign and pre-onset FAR/exposure, exposure detection, effect detection, detected-only latency, precursor and second-stage detection. Confidence intervals resample model seeds first and then paired device clusters, retaining all scenarios and repeated sessions from each device. Single-seed intervals are conditional on that model. Small seed counts limit confidence estimates; zero-event bootstrap intervals are not valid upper-risk guarantees.

## Evaluation and artifacts

Bounded runs use six training, four validation and three test devices per seed, two profiles, 300 samples per ordinary session, plus one-hour benign validation/test histories. Full runs use 12/6/6 devices. All principal stress regimes (weak, startup, slow, intermittent, failed sensors, offsets, environmental drift, spectral blind spots) and four ablations run for every seed. For held-out mechanisms, whole training/calibration sessions containing that family are excluded, including cross-effect labels; test sessions remain intact. Shortcut audits check forbidden columns, device separation and pre-onset scenario predictability. These diagnostics cannot rule out all synthetic shortcuts.

Each immutable run contains `events.json/csv`, `summary.csv`, classification/calibration/rejection reports, operating points, clustered intervals and shortcut audits. `data/` holds local checkpoints, separate Parquet observations/truth and full JSON metadata. The manifest records selected models, schemas, config/source hashes, source snapshots, checksums and exact runtime versions. All artifacts are verified before loading a local trusted checkpoint; checksums provide integrity, not authenticity against an adversary who controls both files and manifest. Replay verifies every detector and regime against saved event rows without modifying the run.

The dashboard refuses missing, stale, corrupt or incompatible runs. Blind source and BB84 sessions keep scenario/onset truth server-side until reveal. The source view labels units/time/reference bands, shows detector branch responsibility and classifier SHAP, and overlays revealed exposure/effect strips. Failed/expired sessions clear the prior display.

## Remaining limits

All results are exploratory synthetic evidence. No frozen external test set or hardware source traces exist. Three-seed bounded intervals are imprecise; longer observation across independent physical devices is required. Family labels compress simultaneous effects. Optional diagnostics and spectral response assumptions materially affect performance. The MDI adapter computes intensity-only observables for two endpoints, not source leakage security; the UI displays one transmitter. Clock-resolved probes, detailed optical waveforms, a source-aware key bound, CV-QKD and hardware intervention validation remain out of scope. The earlier 132/150 result is superseded, not directly comparable after semantic, calibration and simulator changes. Phase 2 historical figures remain preserved and need a full rerun for the corrected decoy blinding model.

## Hardened bounded results (2026-11-28)

Run: `results/phase3/runs/20261128T023314Z-d84d66cb`, seeds 42/43/44, exploratory synthetic evidence. Each profile: 45 nominal attack sessions and 10.2 benign test hours, plus 1.331 pre-onset hours (zero observed pre-onset alarms). The paired receiver histories are identical across profiles.

| Monitor | Exposure hits | Misses | False alarms / benign hours | Detected-only mean latency | Precursor / second-stage hits |
|---|---:|---:|---:|---:|---:|
| Minimal fusion | 35/45 (77.8%) | 10 | 0 / 10.2 h | 24.60 s | 8/9 / 1/9 |
| Instrumented fusion | 36/45 (80.0%) | 9 | 1 / 10.2 h | 35.60 s | 8/9 / 3/9 |
| Receiver-only | 22/45 (48.9%) | 23 | 0 / 10.2 h | 83.06 s | 5/9 / 0/9 |

Seed-then-device 95% bootstrap exposure-detection intervals (500 resamples): minimal 60.0–93.3%, instrumented 66.7–93.3%, receiver 26.7–71.1%. These are imprecise three-seed intervals. A zero-event empirical FAR interval does not establish a zero upper risk bound. Full intervals for latency, FAR and chain stages are in `cluster_intervals.json`.

The old 132/150 result used different semantics, calibration, sensors and exposure; these numbers are not a controlled before/after improvement claim. Persistent-effect hits are separate (minimal 4/45, instrumented 1/45) and are not added to exposure recall. Spectral-blind exposure detection is only 24.4% minimal / 40.0% instrumented; weak attacks 53.3% / 62.2%. Offset-stress FAR rises to 0.74/h / 1.48/h over the shorter stress exposure. Known-sample false rejection ranges across nominal seeds are **51.7–83.6% minimal** and **2.0–66.4% instrumented**. The unknown flag is therefore not a reliable operational novelty detector; high held-out rejection must not be presented without its known-sample error rate.

The full suite passed **83 tests**, with one external-real-data test skipped. Smoke replay reproduced **532 rows** and bounded replay **10,458 rows** exactly. JavaScript syntax and Git whitespace checks passed. Live localhost checks covered missing models, both profiles, blind/reveal, classifier SHAP, keyboard slider movement, mobile-width controls, and the BB84 explicit-reveal flow; no browser errors were observed in those flows. Remote CI is configured but not claimed as executed locally.

Artifacts: `events.json/csv`, `summary.csv`, `classification.json`, `operating_points.json`, `cluster_intervals.json`, `shortcut_audit.json`, `manifest.json` and source snapshots under the run directory. Checkpoints and per-session telemetry remain in its local `data/` directory and are not committed. Smoke run: `results/phase3/runs/20261128T023306Z-15ca2275`. Reproduce with the commands above; replay takes the explicit completed-run path. Historical prototype outputs remain under `results/phase3/full` and `results/phase3/smoke`.
