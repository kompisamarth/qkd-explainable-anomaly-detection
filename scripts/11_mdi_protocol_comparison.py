"""MDI-QKD protocol comparison, detector-attack demonstration, and trial datasets.

Produces:
  results/metrics/mdi_key_rate_vs_distance.csv       MDI vs SNS-TF-QKD vs decoy BB84 vs PLOB
  results/metrics/mdi_detector_attack_comparison.csv Eve owning the measurement device
  results/figures/fig8_mdi_vs_protocols.png
  results/figures/fig9_detector_attack_comparison.png
  data/processed/trial_sessions/trial_class{N}.csv   one replayable session per attack class

The trial sessions are generated with randomize_nuisance=True, i.e. from the same
leakage-free distribution the deployed model was trained on, and carry the model's
per-block predictions so the dashboard can replay them instantly.
"""

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.features.time_series import FEATURE_NAMES_63, CausalMultiAttackFeatureExtractor
from src.simulator.bb84_engine import BB84Simulator
from src.simulator.mdi_qkd import (
    MDIChannelParams,
    MDIQKD,
    detector_attack_comparison,
)
from src.simulator.tf_qkd import ChannelParams, SNSTFQKD, decoy_bb84_key_rate, plob_bound

CLASS_SHORT = ["Normal", "Fractional IR", "PNS Attack", "Fake State", "Time-Shift"]
METRICS = ROOT_DIR / "results" / "metrics"
FIGURES = ROOT_DIR / "results" / "figures"


def key_rate_curves():
    print("[1/3] MDI-QKD key rate vs distance (with TF-QKD / BB84 / PLOB comparison)...")
    mdi_ch = MDIChannelParams()
    mdi = MDIQKD(mdi_ch)
    tf_ch = ChannelParams()
    sns = SNSTFQKD(tf_ch)

    distances = np.arange(0, 451, 25)
    rows = []
    for d in distances:
        dd = max(float(d), 1e-3)
        r_mdi = mdi.optimize(dd)[1]["key_rate"]
        r_sns = sns.optimize(dd)[1]["key_rate"]
        r_bb84 = decoy_bb84_key_rate(dd, tf_ch)[0]
        rows.append({
            "distance_km": float(d),
            "mdi_qkd": r_mdi,
            "sns_tf_qkd": r_sns,
            "decoy_bb84": r_bb84,
            "plob_bound": plob_bound(dd, tf_ch),
        })
        print(f"   {d:3.0f} km  MDI={r_mdi:.3e}  SNS-TF={r_sns:.3e}  BB84={r_bb84:.3e}")

    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "mdi_key_rate_vs_distance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    for col, lbl, style in [
        ("plob_bound", "PLOB repeaterless bound", "k--"),
        ("sns_tf_qkd", "SNS-TF-QKD (sqrt-eta)", "-o"),
        ("mdi_qkd", "MDI-QKD (eta, detector-secure)", "-s"),
        ("decoy_bb84", "Decoy-state BB84 (eta)", "-^"),
    ]:
        v = df[col].replace(0, np.nan)
        ax.semilogy(df["distance_km"], v, style, label=lbl, ms=3.5, lw=1.6)
    ax.set_xlabel("Distance (km)", fontweight="bold")
    ax.set_ylabel("Secret key rate (bits / pulse)", fontweight="bold")
    ax.set_title("Key rate vs distance: MDI-QKD trades rate for detector security",
                 fontweight="bold", fontsize=11)
    ax.grid(True, which="both", alpha=0.3, ls="--")
    ax.legend(fontsize=8)
    ax.set_ylim(1e-11, 1)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_mdi_vs_protocols.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return df


def attack_comparison():
    print("\n[2/3] Detector attack: Eve owns the measurement device...")
    rows = detector_attack_comparison(p_grid=np.linspace(0, 1, 21))
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "mdi_detector_attack_comparison.csv", index=False)

    full = df.iloc[-1]
    print(f"   At full control (p=1.0):")
    print(f"     BB84  : Eve info {full['bb84_eve_information']:.0%}, "
          f"observed QBER {full['bb84_qber']:.2%}, check-basis error "
          f"{full['bb84_check_error']:.2%}, aborts={bool(full['bb84_aborts'])}")
    print(f"     MDI   : Eve info {full['mdi_eve_information']:.0%}, "
          f"observed QBER {full['mdi_qber']:.2%}, phase error "
          f"{full['mdi_check_error']:.2%}, aborts={bool(full['mdi_aborts'])}")
    first_abort = df.loc[df["mdi_aborts"], "p_control"].min()
    print(f"   MDI-QKD trips its monitor at p_control = {first_abort:.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=300)
    ax1.plot(df["p_control"], df["bb84_check_error"] * 100, "-o", color="#EF4444",
             ms=3.5, label="BB84: error Eve lets you see")
    ax1.plot(df["p_control"], df["mdi_check_error"] * 100, "-s", color="#2563EB",
             ms=3.5, label="MDI-QKD: X-basis phase error")
    ax1.axhline(11, color="k", ls="--", lw=1.4, label="11% abort threshold")
    ax1.set_xlabel("Fraction of the measurement device Eve controls", fontweight="bold")
    ax1.set_ylabel("Security monitor (%)", fontweight="bold")
    ax1.set_title("The monitor Eve cannot forge", fontweight="bold", fontsize=10)
    ax1.grid(True, alpha=0.3, ls="--")
    ax1.legend(fontsize=8)

    ax2.plot(df["p_control"], df["bb84_eve_information"] * 100, "-o", color="#EF4444",
             ms=3.5, label="BB84 (key stolen silently)")
    ax2.plot(df["p_control"], df["mdi_eve_information"] * 100 * (1 - df["mdi_aborts"]),
             "-s", color="#2563EB", ms=3.5, label="MDI-QKD (after abort logic)")
    ax2.set_xlabel("Fraction of the measurement device Eve controls", fontweight="bold")
    ax2.set_ylabel("Key bits Eve retains (%)", fontweight="bold")
    ax2.set_title("Information Eve actually keeps", fontweight="bold", fontsize=10)
    ax2.grid(True, alpha=0.3, ls="--")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig9_detector_attack_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return df


def trial_sessions():
    print("\n[3/3] Generating replayable trial sessions (leakage-free distribution)...")
    ckpt = ROOT_DIR / "data" / "processed" / "checkpoints" / "xgb_multiattack.joblib"
    model = joblib.load(ckpt) if ckpt.exists() else None
    if model is None:
        print("   WARNING: no model checkpoint; trial sessions will omit predictions.")

    out_dir = ROOT_DIR / "data" / "processed" / "trial_sessions"
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = CausalMultiAttackFeatureExtractor(window_size=50, ema_span=10)
    summary = []
    for cls in range(5):
        sim = BB84Simulator(raw_photons_per_block=1024, random_state=2000 + cls)
        seq = sim.simulate_session(
            session_id=cls,
            target_attack_class=cls,
            channel_type="fiber",
            blocks_per_session=150,
            attack_start_block=76,
            attack_mode="multiattack",
            randomize_nuisance=True,
        )
        df = extractor.extract_from_session(seq)

        if model is not None:
            probs = model.predict_proba(df[FEATURE_NAMES_63].values)
            df["pred_class"] = probs.argmax(axis=1)
            for k in range(5):
                df[f"prob_class_{k}"] = probs[:, k]

            post = df[df["block_index"] >= 76]
            correct = float((post["pred_class"] == cls).mean()) if len(post) else float("nan")
            alarms = post[post["pred_class"] != 0]["block_index"]
            delta_t = int(alarms.min() - 76) if len(alarms) else -1
        else:
            correct, delta_t = float("nan"), -1

        path = out_dir / f"trial_class{cls}.csv"
        df.to_csv(path, index=False)
        summary.append({
            "class_id": cls,
            "class_name": CLASS_SHORT[cls],
            "p_eve": float(seq.p_eve),
            "blocks": len(df),
            "post_onset_accuracy": correct,
            "detection_latency_blocks": delta_t,
            "file": path.name,
        })
        print(f"   class {cls} ({CLASS_SHORT[cls]:14s}) p_eve={seq.p_eve:.3f} "
              f"post-onset acc={correct:.3f} dT={delta_t}")

    sdf = pd.DataFrame(summary)
    sdf.to_csv(METRICS / "trial_session_summary.csv", index=False)
    return sdf


def main():
    METRICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("MDI-QKD PROTOCOL COMPARISON, DETECTOR ATTACK DEMO, AND TRIAL DATA")
    print("=" * 78)
    key_rate_curves()
    attack_comparison()
    trial_sessions()
    print("\n" + "=" * 78)
    print("DONE. Wrote metrics, figures 8-9, and data/processed/trial_sessions/.")
    print("=" * 78)


if __name__ == "__main__":
    main()
