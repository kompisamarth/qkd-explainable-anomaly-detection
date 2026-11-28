# Phase 3 hardening audit — 2026-11-28

Scope: local Phase 3 implementation and its BB84/dashboard integration. The initial eight hardening regressions failed against the prototype; all passed after correction. Two additional pre-fix tests confirmed BB84 blind disclosure and intensity-dependent blinding trigger efficiency. Historical artifacts are preserved. No hardware measurements or security-proof validation were performed.

## Evidence ledger

Statuses refer to the code and tests, not to scientific validation of assumed parameters.

| Finding | Status and evidence |
|---|---|
| 1. Overlap credited as detection | Resolved: shared `event_metrics` requires a rising alarm during exposure; onset-crossing regression. |
| 2. Warm-up creates artificial alarm | Resolved: transitions computed before warm-up masking; startup has explicit zero-time scoring; boundary tests. |
| 3. Mixed-score maximum | Resolved: validation-benign median/95th excursion normalization plus separately selected fusion cutoff; no ECDF tail saturation. |
| 4. Profile-dependent sensor RNG | Resolved: stable sensor identity seeds; identical shared-sensor frames across profiles. |
| 5. Lost saturation/stale data | Resolved: clipping held with sample; explicit staleness; health includes stale/missing/clipped. |
| 6. Uncalibrated probabilities | Resolved in implementation: temperature calibration on separate validation devices; held-out ECE/Brier/log loss reported. Calibration quality remains empirical. |
| 7. Receiver comparator at different operating point | Resolved: receiver training normalization and common threshold/event procedure. Achieved rates need not match exactly. |
| 8. Unsupported environment | Resolved: isolated Python 3.13.13, NumPy 2.1.3, exact Phase 3 lock; full suite executed there. |
| 9. Hidden numerical assumptions | Main experiment-defining mappings moved to versioned source config; model discretization/order documented. Numerical calibration remains synthetic. |
| 10. Family labels conceal effects | Resolved: requested mechanism, incident power, wavelength and simultaneous component flags exported separately. Single-label classifier remains a simplification. |
| 11. Chain cross-effects | Resolved in semantics: explicit second-stage wavelength/power; effects inherited, not artificially removed; chain-stage alarms scored independently. |
| 12. Invalid config/schedule | Guards added for cadence, probability, sensor bounds, positive dynamics, schedule consistency, repair and chain ordering; invalid-input regressions. |
| 13. Cadence-dependent feature windows | Resolved: reject irregular/nonfinite/config-inconsistent timestamps; seconds used for slopes. |
| 14. Instantaneous ablation retained history | Resolved: current values/residuals only; no age, stale, quality, delta or rolling terms. |
| 15. Startup bypassed sensor response | Resolved: nominal unexposed filter initial state, then configured response. |
| 16. Fixed receiver counts | Resolved: effective sampled pulse rate times cadence; counts exported and tested. |
| 17. QBER with no detections | Resolved: null/NaN, never an invented zero/error bit; zero-click regression. |
| 18. No shortcut diagnostics | Added allowlist/device-disjoint checks and pre-onset scenario-prediction diagnostic. This does not prove absence of every simulator shortcut. |
| 19. Short benign exposure | Added one-hour independent benign sessions per validation/test device in bounded/full runs. More physical exposure is still needed. |
| 20. Single-seed stress/ablations | Resolved: all principal regimes and four ablations on every requested seed. |
| 21. Replay/evaluator mismatch | Resolved: replay imports the same evaluator and compares every saved event row exactly. |
| 22. Pre-onset FAR omitted | Resolved: separate pre-onset exposure, event counts and rates. |
| 23. Single unknown class / fixed 0.6 | Added separately held-out seeding/photo with whole-session exclusion, validation-derived confidence cutoff, known false rejection and unknown rejection. No universal novelty guarantee. |
| 24. Missing uncertainty/dependence | Added seed/model then device-cluster bootstrap for detection, FAR, detected latency and both chain stages. Few seeds/zero-FAR limitations explicit. |
| 25. Mixed rerun output | Resolved: unique run directories, manifest completion marker, source-change guard, atomic local pointer publication. |
| 26. Lexical checkpoint selection | Resolved: explicit deployment mapping in validated manifest. |
| 27. Metadata lost on export | Resolved: full schedule/device/sensor/config/offset/gain metadata plus observations/truth; replay uses it. |
| 28. Dirty commit insufficient provenance | Resolved: source-content hashes and snapshots, config hashes, schemas, checksums and runtime versions. |
| 29. Missing CI/reproduce integration | Added Phase 3 CI smoke+replay and `reproduce.py --phase3 --quick`. Remote CI execution is distinct from local verification. |
| 30. Optional RF extension | Deferred: primary classifier is XGBoost; Isolation Forest is the independent anomaly branch. RF comparison is not claimed. |
| 31. SHAP misattributed fusion | Resolved: classifier-margin label and dominant fusion branch displayed separately. |
| 32. Chart units/time/bands | Added physical-time ticks, sensor units, normalized-intensity ratio and explicit reference-band meaning. |
| 33. Reveal only metadata | Resolved: revealed exposure, mechanism stage and effect strips; shared event metrics shown. |
| 34. Stale UI/async races | Resolved: failed sessions clear; response/session/sample identities guard late responses; unavailable controls disabled. |
| 35. Repeated model/explainer loading | Added bounded model cache and lazy explainer cache; artifact verification remains mandatory. |
| 36. Checkpoint compatibility | Resolved: manifest/schema/runtime/source/hash validation and explicit errors. Checksums are integrity checks, not authenticity. |
| 37. Source iframe layout | Moved inside main content wrapper; fluid cards/labels and keyboard controls. Live checks recorded below. |
| 38. Stale running backend | Restarted normal localhost:8000 with pinned interpreter and current code; stopped old 8001 process. |
| 39. BB84 blind disclosure | Resolved: server randomness, bounded server-side sealed truth, explicit reveal endpoint; response regression and live reveal flow. |
| 40. Decoy checkpoint mismatch | Base 63-feature checkpoint capability explicitly returned and schema checked. Decoy-aware deployment remains unavailable; no such capability claimed. |
| 41. Blinding decoy trigger | Resolved: single signal-calibrated trigger efficiency reused for every intensity, vacuum included; analytic ratio regression. Historical Phase 2 decoy figures require rerun. |
| 42. Finite latency for misses / detection definitions | Misses now NaN/null; pre-onset alarms unaffected by correct-class mask. API exposes both any-alarm and correct-class latency; UI labels distinction. Historical CSVs preserved. |
| 43. MDI claims too strong | Analytic mixture explicitly labeled; removed advance-basis knowledge claim. No arbitrary-adversary/security-proof validation claimed. |
| 44. Stale README and threat matrix | Rewritten around current implementation, explicit proof assumptions, calibrated event semantics, reproduction and unresolved limits. |

## Verification

- Initial regression run: 8 failures exposed event, RNG, sensor, cadence, schedule and missed-latency defects.
- Integration regression run: 2 additional failures exposed blind disclosure and decoy trigger inconsistency.
- Pinned-environment full suite: **83 passed, 1 skipped**, 34.07 s. Skip: external Zenodo receiver telemetry absent. One FastAPI/Starlette httpx deprecation warning; no failed test.
- Smoke run `20261128T023306Z-15ca2275`: completed; **532 event rows replayed exactly**, with hashes/schemas/runtime verified.
- Live browser: normal entry point opened, missing-model state disabled simulation and explanation, BB84 blind view hid onset-dependent metrics, explicit reveal worked, no browser errors in that flow.
- Additional bounded-run results and final live profile checks are recorded below after completion.

## Scientific limitations

These are exploratory simulations. Parameter mappings and optional sensors are assumptions, probability calibration may fail under shifts, held-out mechanisms are only two selected cases, and three model seeds provide limited uncertainty resolution. Pure-benign and pre-onset rates should be interpreted separately. A persistent alarm can be operationally useful yet miss a new event under the strict evaluator. Source effects are not tied to a composable key bound. No source hardware validation, frozen external test, clock-resolved watchdog, full two-endpoint telemetry dashboard or RF benchmark is claimed. Historical Phase 2 numerical tables are not regenerated by this bounded hardening task.

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
