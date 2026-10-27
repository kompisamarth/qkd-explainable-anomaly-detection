"""Modern QKD: Sending-or-Not-Sending Twin-Field QKD (SNS-TF-QKD) vs. decoy-state BB84.

Generates:
- Figure 6: Asymptotic secret-key rate vs. distance for SNS-TF-QKD, efficient decoy-state
  BB84 and the repeaterless PLOB bound.
- Figure 7: Single-session trace at 200 km with a phase-tampering attack at block 101,
  showing that the Z-window bit error rate is blind to it while the X-window phase error rises.
- Metrics: key-rate table, and detection latency of a Z-QBER monitor, a static threshold and
  Hoeffding test on the phase error, and the temporal ML detector re-used from the BB84 pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.simulator.tf_qkd import (
    ChannelParams, SNSIntensities, SNSTFQKD, SNSTimeSeriesSimulator,
    decoy_bb84_key_rate, plob_bound,
)
from src.features.time_series import FEATURE_NAMES, CausalTimeSeriesExtractor
from src.models.hoeffding_baseline import HoeffdingDetector
from src.models.ml_classifiers import QKDMLDetector


def load_config() -> dict:
    with open(ROOT_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def key_rate_sweep(model: SNSTFQKD, distances: np.ndarray, e_d_bb84: float) -> pd.DataFrame:
    rows, prev = [], None
    for d in distances:
        x_opt, st = model.optimize(float(d), prev)
        prev = x_opt
        r_bb84, mu_bb84 = decoy_bb84_key_rate(float(d), model.ch, e_d=e_d_bb84)
        rows.append({
            "distance_km": float(d),
            "sns_tf_key_rate": st["key_rate"],
            "decoy_bb84_key_rate": r_bb84,
            "plob_bound": plob_bound(float(d), model.ch),
            "sns_E_Z": st["E_Z"], "sns_e1_phase": st["e1_phase_upper"],
            "sns_eps": x_opt.eps, "sns_mu_z": x_opt.mu_z, "sns_mu_1": x_opt.mu_1, "sns_mu_2": x_opt.mu_2,
            "bb84_mu": mu_bb84,
        })
    return pd.DataFrame(rows)


def crossing_distance(df: pd.DataFrame, a: str, b: str) -> float:
    """First distance at which column a exceeds column b (both positive)."""
    ok = (df[a] > df[b]) & (df[a] > 0)
    return float(df.loc[ok, "distance_km"].iloc[0]) if ok.any() else float("nan")


def plot_key_rates(df: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(9, 5.5), dpi=300)
    for col, label, style in [
        ("plob_bound", "PLOB bound (repeaterless limit)", dict(color="black", ls="--")),
        ("decoy_bb84_key_rate", "Decoy-state BB84 (efficient, asymptotic)", dict(color="#1f77b4")),
        ("sns_tf_key_rate", "SNS Twin-Field QKD (asymptotic)", dict(color="#d62728")),
    ]:
        sub = df[df[col] > 0]
        plt.semilogy(sub["distance_km"], sub[col], label=label, lw=2.2, **style)
    plt.xlabel("Fiber distance Alice-Bob (km)", fontweight="bold")
    plt.ylabel("Secret key rate (bits / pulse)", fontweight="bold")
    plt.title("Twin-Field QKD beats the repeaterless bound", fontweight="bold")
    plt.ylim(1e-11, 1)
    plt.grid(True, which="both", alpha=0.4, ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig6_tfqkd_key_rate_vs_distance.png", dpi=300, bbox_inches="tight")
    plt.close()


def session_to_features(extractor, sid, ses, p_eve, attack_start):
    df = extractor.extract_from_session(
        session_id=sid, qbers=ses.phase_err_x, sifted_lens=ses.n_x_events,
        block_classes=ses.block_classes, attack_class=2 if p_eve > 0 else 0,
        channel_type="tf_fiber", p_eve=p_eve, attack_start_block=attack_start,
    )
    df["qber_z"] = ses.qber_z[df["block_index"].values - 1]
    return df


def build_dataset(sim, extractor, n_sessions, p_range, attack_start, blocks, sid0, rng):
    dfs = []
    for i in range(n_sessions):
        attack = i % 2 == 1
        p = float(rng.uniform(*p_range)) if attack else 0.0
        ses = sim.simulate_session(attack, p, blocks, attack_start)
        dfs.append(session_to_features(extractor, sid0 + i, ses, p, attack_start))
    return pd.concat(dfs, ignore_index=True)


def latency_table(sim, extractor, detectors, p_grid, n_per_p, attack_start, blocks, seed):
    rows = []
    for p in p_grid:
        sim.rng = np.random.default_rng(seed + int(p * 1e4))   # same sessions for every detector
        sessions = [session_to_features(extractor, s, sim.simulate_session(True, p, blocks, attack_start), p, attack_start)
                    for s in range(n_per_p)]
        for name, det in detectors.items():
            lat, hit, fa, pre, post_alarms, post = [], [], 0, 0, 0, 0
            for df in sessions:
                pred = np.asarray(det(df))
                bi = df["block_index"].values
                fa += int(pred[bi < attack_start].sum())
                pre += int((bi < attack_start).sum())
                alarms = bi[(bi >= attack_start) & (pred == 1)]
                post_alarms += len(alarms)
                post += int((bi >= attack_start).sum())
                hit.append(len(alarms) > 0)
                lat.append(int(alarms[0] - attack_start) if len(alarms) else blocks - attack_start + 1)
            rows.append({"p_eve": p, "model": name, "mean_delta_t": float(np.mean(lat)),
                         "detection_rate": float(np.mean(hit)), "pre_attack_fpr": fa / max(pre, 1),
                         # Alarm rate on attacked blocks; equal to pre_attack_fpr means the detector is blind
                         "post_attack_alarm_rate": post_alarms / max(post, 1),
                         "num_sessions": n_per_p})
    return pd.DataFrame(rows)


def plot_session(ses, ml_det, extractor, attack_start, fig_dir):
    df = session_to_features(extractor, 0, ses, ses.p_eve, attack_start)
    probs = ml_det.predict_proba(df)[:, 1]
    blocks = np.arange(1, len(ses.qber_z) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, dpi=300)
    axes[0].plot(blocks, ses.qber_z * 100, color="#8c564b", alpha=0.8, label="Z-window bit error rate $E_Z$")
    axes[0].set_ylabel("$E_Z$ (%)", fontweight="bold")
    axes[0].set_title(f"SNS-TF-QKD at 200 km: phase tampering on {ses.p_eve:.0%} of pulses from block {attack_start}", fontweight="bold")
    axes[1].plot(blocks, ses.phase_err_x * 100, color="#1f77b4", alpha=0.6, label="X-window phase error $E_X$")
    axes[1].plot(df["block_index"], df["rolling_mean"] * 100, color="#1f77b4", lw=2.5, label="Rolling mean (W=10)")
    axes[1].set_ylabel("$E_X$ (%)", fontweight="bold")
    axes[2].plot(df["block_index"], probs, color="#d62728", lw=2, label="ML posterior P(attack)")
    axes[2].axhline(ml_det.calibrated_threshold, color="black", ls=":", label="Calibrated threshold")
    axes[2].set_ylabel("P(attack)", fontweight="bold")
    axes[2].set_xlabel("Block index", fontweight="bold")
    axes[2].set_ylim(-0.05, 1.05)
    for ax in axes:
        ax.axvline(attack_start, color="#2ca02c", lw=2)
        ax.axvspan(attack_start, blocks[-1], color="#e8f5e9", alpha=0.5)
        ax.grid(True, alpha=0.4, ls="--")
        ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig7_tfqkd_phase_attack_trace.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 70)
    print("MODERN QKD: SENDING-OR-NOT-SENDING TWIN-FIELD QKD")
    print("=" * 70)
    cfg = load_config()
    tcfg = cfg["tf_qkd"]
    metrics_dir = ROOT_DIR / cfg["paths"]["metrics"]
    fig_dir = ROOT_DIR / cfg["paths"]["figures"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["simulation"]["random_seed"]

    ch = ChannelParams(**tcfg["channel"])
    model = SNSTFQKD(ch)

    # 1. Key rate vs distance
    print("\n[1/3] Optimizing key rates vs. distance...")
    df_rate = key_rate_sweep(model, np.arange(10, 701, 10), tcfg["bb84_misalignment"])
    df_rate.to_csv(metrics_dir / "tfqkd_key_rate_vs_distance.csv", index=False)
    plot_key_rates(df_rate, fig_dir)
    for d in [50, 100, 200, 300, 400]:
        r = df_rate[df_rate["distance_km"] == d].iloc[0]
        print(f"  {d:4d} km | SNS-TF: {r.sns_tf_key_rate:.3e} | Decoy BB84: {r.decoy_bb84_key_rate:.3e} | PLOB: {r.plob_bound:.3e}")
    max_d = lambda c: float(df_rate.loc[df_rate[c] > 0, "distance_km"].max())
    print(f"  SNS-TF overtakes decoy BB84 at ~{crossing_distance(df_rate, 'sns_tf_key_rate', 'decoy_bb84_key_rate'):.0f} km")
    print(f"  SNS-TF beats the PLOB bound from ~{crossing_distance(df_rate, 'sns_tf_key_rate', 'plob_bound'):.0f} km")
    print(f"  Max distance: SNS-TF {max_d('sns_tf_key_rate'):.0f} km | decoy BB84 {max_d('decoy_bb84_key_rate'):.0f} km")
    print("  Saved fig6_tfqkd_key_rate_vs_distance.png")

    # 2. Time-series sessions and detectors
    print("\n[2/3] Simulating finite-block sessions with phase drift and phase tampering...")
    ts = tcfg["time_series"]
    x = SNSIntensities(**ts["intensities"])
    st = model.statistics(ts["distance_km"], x)
    print(f"  Operating point {ts['distance_km']} km: E_Z={st['E_Z']:.4f}, E_X={st['E_X']:.4f}, "
          f"e1_ph<={st['e1_phase_upper']:.4f}, R={st['key_rate']:.3e} bits/pulse")
    sim = SNSTimeSeriesSimulator(
        model, ts["distance_km"], x, ts["z_pairs_per_block"], ts["x_pairs_per_block"],
        ts["phase_drift_std"], ts["phase_drift_phi"], random_state=seed,
    )
    extractor = CausalTimeSeriesExtractor(cfg["features"]["window_size"], cfg["features"]["ema_span"])
    rng = np.random.default_rng(seed)
    blocks, a0, p_range = ts["blocks_per_session"], ts["attack_start_block"], tuple(ts["p_eve_range"])
    train = build_dataset(sim, extractor, ts["train_sessions"], p_range, a0, blocks, 0, rng)
    val = build_dataset(sim, extractor, ts["val_sessions"], p_range, a0, blocks, 10_000, rng)

    ml = QKDMLDetector("random_forest", "binary", target_fpr=0.01, model_params=cfg["models"]["random_forest"])
    ml.fit(train[FEATURE_NAMES], train["is_attack_active"].values, val[FEATURE_NAMES], val["is_attack_active"].values)
    benign_ex = train.loc[train["is_attack_active"] == 0, "qber_current"].mean()
    benign_ez = train.loc[train["is_attack_active"] == 0, "qber_z"]
    hoeff = HoeffdingDetector(base_qber=float(benign_ex), alpha=0.01, mode="windowed", window_size=extractor.window_size)
    ez_thresh = float(np.percentile(benign_ez, 99.0))
    print(f"  Benign mean E_X = {benign_ex:.4f}; Z-QBER monitor threshold (99th pct benign) = {ez_thresh:.4f}")

    detectors = {
        "Z-QBER monitor (99th pct)": lambda d: (d["qber_z"] >= ez_thresh).astype(int).values,
        "Static 11% on E_X": lambda d: (d["qber_current"] >= 0.11).astype(int).values,
        "Hoeffding windowed on E_X": hoeff.predict,
        "Random Forest on E_X features": ml.predict,
    }
    df_lat = latency_table(sim, extractor, detectors, ts["latency_p_eve_grid"], ts["latency_sessions_per_p"], a0, blocks, seed)
    df_lat.to_csv(metrics_dir / "tfqkd_detection_latency.csv", index=False)
    print(df_lat.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 3. Trace figure
    print("\n[3/3] Plotting single-session trace...")
    sim.rng = np.random.default_rng(seed + 777)
    plot_session(sim.simulate_session(True, ts["trace_p_eve"], blocks, a0), ml, extractor, a0, fig_dir)
    print("  Saved fig7_tfqkd_phase_attack_trace.png")

    print("\n" + "=" * 70)
    print("TF-QKD SIMULATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
