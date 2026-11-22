# Limitations

Read this before quoting any number from `results/`.

## 1. Simulated data
Every training and test row, and every benchmark in `results/metrics/` except the files
prefixed `real_`, comes from the simulators in `src/simulator/`. The only real measurements
used anywhere are the benign Zenodo 21792844 telemetry, and only for a calibration check and an
out-of-distribution test. That test shows the simulation-trained model **does not transfer**:
it flags every real benign block as an attack, because the real system's baseline QBER (≈ 3.3%
at 10 dB) is higher than the simulator's (1.6–2.2%).

## 2. Simplified attack models
All six attacks are statistical caricatures of the hardware attacks (`docs/ATTACK_MODELS.md`):
* Eve's detectors and quantum memory are ideal.
* She tunes her attack to match Bob's click rate with a single relative error.
* There is no timing jitter or electronics, and no optical-power telemetry.
* The Trojan-horse model only adds noise at Bob.
* Time-shift uses a two-level efficiency step.

Real attacks leave other traces (photocurrent, temperature, arrival-time histograms) that this
telemetry does not contain, and may leave fewer of the traces modelled here.

## 3. Asymptotic vs finite-key assumptions
* The 11% abort level is the asymptotic one-way BB84 bound (Shor & Preskill 2000). It is used only
  as a baseline.
* Per-block QBER carries a Wilson 95% interval, and a finite-size threshold variant is reported.
  No finite-key secret-key-rate analysis is performed.
* The TF-QKD, MDI-QKD and decoy-BB84 key rates are asymptotic (infinite decoys; no finite-size
  corrections).
* The MDI detector-attack comparison is an analytic illustration, not a simulation.

## 4. No hardware validation
No result here was produced or checked on QKD hardware, and the attack models were never
validated against a real attack. The QpiAI extras run on the NumPy fallback unless an API key is
configured. Even then, the local QpiAI QRNG is pseudo-random; only a QPU gives quantum randomness.

## 5. What the numbers do and do not mean
* Accuracy and F1 describe this simulator under `config/physics.yaml`. They are not a property
  of real QKD systems.
* Three of the seven classes (PNS without decoys, Trojan horse, beam splitting) are designed to
  be invisible in Bob's QBER telemetry. Low recall on them is the correct physical outcome, and
  it lowers every aggregate metric.
* Held-out results (unseen intensities and 80–110 km) show a large rise in false alarms: the
  model does not extrapolate to longer distances.
* Per-block labels mark a window as attacked once its *last* block is attacked, so windows that
  straddle the onset are partly benign. This lowers accuracy and lengthens latency.
* SVM-RBF trains on a session-stratified subsample (≤ 12,000 rows) because its cost grows
  quadratically. The feature-group ablation is single-seed.
