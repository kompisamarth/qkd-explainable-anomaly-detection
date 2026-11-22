# Attack Models

Every attack in `src/simulator/calibrated.py` is a **simplified, statistical** model of a
hardware attack. None of them simulates optics, electronics or timing jitter. Each acts
on the click and error probabilities of pulse populations, and Eve is assumed to tune
her attack to hide from Bob where the real attack allows it. Parameters and their
sources are in `config/physics.yaml` (`calibrated_bb84.attacks`).

"Intensity" is the one strength knob sampled per session. Training uses
`intensity`; the generalisation test set uses the disjoint `heldout_intensity`.

| Class | Model | Intensity | What Bob can observe | Real attack | Main simplifications |
|---|---|---|---|---|---|
| 1 Intercept-resend | Eve measures a fraction of pulses in a random basis (perfect detector, so only vacuum pulses escape her) and resends a coherent pulse in her basis and bit. With `gain_matching` she picks the resend intensity so Bob's signal click rate is unchanged. | Fraction intercepted | QBER rises by ≈ f/4 | Bennett & Brassard 1984; Bennett et al., J. Cryptology 5, 3 (1992) | Perfect detector for Eve; one resend intensity for every pulse; no timing signature |
| 2 PNS | Quantum non-demolition photon counting. Single photons are blocked (or partly forwarded); multi-photon pulses lose one photon to Eve and the rest travel losslessly to Bob. Eve tunes forwarding to match Bob's *signal* click rate, with a per-session relative error `gain_mismatch`. | Fraction of pulses she acts on | No QBER change. Without decoys, only the gain mismatch remains; with decoys, decoy and vacuum yields disagree with the signal (`y1_lower_bound_ratio` drops) | Brassard, Lütkenhaus, Mor & Sanders, PRL 85, 1330 (2000) | Ideal QND measurement and quantum memory; Eve sees photon number but not the intensity label |
| 3 Detector blinding | Bright light keeps Bob's detectors in linear mode. Eve intercepts, then sends trigger pulses that click only when Bob's basis equals hers, giving her bit with a residual error. Blinded detectors produce no dark counts. | Fraction of time blinded | QBER *falls* below the optical floor e_d; dark-count floor disappears at long distance | Lydersen et al., Nat. Photon. 4, 686 (2010) | No photocurrent or temperature telemetry; trigger click probability tuned to match gain |
| 4 Time-shift | Eve shifts arrival times so one detector of each pair is less efficient (ratio r), early or late at random. Photons are not measured or resent. | Fraction of pulses shifted | Click rate drops by up to (1 − r)/2 of the shifted fraction; QBER essentially unchanged | Qi et al., QIC 7, 73 (2007); Zhao et al., PRA 78, 042333 (2008) | Detector efficiency is a two-level step function of time; Eve's information approximated as f(1 − r)/(1 + r) |
| 5 Trojan horse | Eve injects light into Alice's modulator and reads the back-reflection. Bob's only signature is a small leak of Eve's light reaching his detectors as extra noise. | Mean back-reflected photon number | Tiny extra noise clicks; in reality Alice's watchdog photodiode sees it (recorded as `alice_monitor`, never a model feature) | Gisin et al., PRA 73, 022320 (2006); Lucamarini et al., PRX 5, 031030 (2015) | No modulator model; leak ratio is an assumption; Eve's information is 1 − exp(−intensity) |
| 6 Beam splitting | Eve taps a fraction of every pulse and (with `loss_compensation`) replaces the fibre by a lower-loss link so Bob's received intensity is unchanged. | Tapped fraction | Nothing, while the tap can be compensated; at very short links the gain drops | Bennett et al. (1992); Huttner et al., PRA 51, 1863 (1995) | Passive tap only; ignores coherent attacks on the tapped light |

## Expected behaviour (what an honest result looks like)

Trojan horse and beam splitting are designed to leave Bob's QBER telemetry statistically
unchanged, and PNS is too without decoy states. A classifier that reaches high accuracy
on those classes from QBER features alone would indicate leakage, not skill. Low recall
on them is the physically correct outcome; the decoy-state experiment shows how much PNS
detection improves when the decoy features are added.

## Eve's information

`eve_info_fraction` (dataset metadata, not a feature) is a rough per-block estimate of
the fraction of Bob's sifted bits Eve learns. It is used only to describe severity and is
never a model input.
