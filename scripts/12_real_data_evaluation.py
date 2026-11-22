"""Real QKD telemetry (Zenodo 21792844): simulator calibration check and OOD false-alarm test.

Outputs (all labelled as REAL data, kept apart from simulated results):
  results/metrics/real_vs_sim_qber_calibration.csv   real mean QBER vs attenuation next to the
                                                      simulator's benign QBER at the same loss
  results/metrics/real_data_ood_false_alarms.csv     per real block: model prediction (every
                                                      real block is benign, so any attack is a false alarm)
  results/metrics/real_data_ood_summary.json
  results/figures/fig_real_vs_sim_qber.png

    python scripts/12_real_data_evaluation.py              # uses files already in data/raw
    python scripts/12_real_data_evaluation.py --download   # first fetch all 39 files (~2.2 MB)
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import load_config
from src.data.real_telemetry import (CITATION, ORDER_FREE_FEATURES, download, load_blocks,
                                     summary_qber_vs_attenuation)
from src.simulator.calibrated import CLASS_NAMES, CalibratedBB84Simulator

METRICS = ROOT_DIR / "results" / "metrics"
FIGURES = ROOT_DIR / "results" / "figures"
REAL_LABEL = "Real data (benign, Zenodo 21792844)"


def simulated_qber_vs_loss(phys, losses_db, sessions=6, seed=7):
    """Benign simulated QBER at fixed total loss, per detector type."""
    rows = []
    for det in phys["detector"]["types"]:
        p = copy.deepcopy(phys)
        p["detector"]["types"] = [det]
        sim = CalibratedBB84Simulator(p, random_state=seed)
        for loss in losses_db:
            q = []
            for s in range(sessions):
                rng = np.random.default_rng(seed * 1000 + s)
                sp = sim.draw_session_physics(rng, "normal", blocks=40)
                sp.loss_db, sp.loss_step_block = float(loss), -1
                seq = sim.simulate_session(s, 0, blocks_per_session=40, attack_start_block=41,
                                           session_seed=seed * 1000 + s, physics=sp)
                q.append(seq.block_errors.sum() / max(seq.block_sifted_lens.sum(), 1))
            rows.append({"detector": det, "loss_db": float(loss), "sim_mean_qber": float(np.mean(q)),
                         "sim_std_qber": float(np.std(q, ddof=1))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="fetch the full Zenodo record first")
    ap.add_argument("--raw-dir", default=str(ROOT_DIR / "data" / "raw"))
    args = ap.parse_args()
    raw = Path(args.raw_dir)
    if args.download:
        files = download(raw / "zenodo_21792844")
        print(f"Downloaded {len(files)} files to {raw / 'zenodo_21792844'}")

    cfg = load_config()
    phys = cfg["calibrated_bb84"]
    METRICS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("REAL QKD TELEMETRY: CALIBRATION CHECK AND OOD FALSE-ALARM TEST")
    print(f"  Source: {CITATION}")
    print("=" * 74)

    # ---- 1. Calibration: real QBER vs attenuation, next to the simulator ----
    real_sum = summary_qber_vs_attenuation(raw)
    if real_sum.empty:
        print("  No *_stats.json summaries found; skipping calibration.")
    else:
        # The summaries report VOA settings; the 50 km files add ~10 dB of fibre (0.2 dB/km).
        real_sum["fibre_db"] = np.where(real_sum["source"].str.contains("50km"), 10.0, 0.0)
        real_sum["total_loss_db"] = real_sum["voa_db"] + real_sum["fibre_db"]
        # Only the no-injection baseline runs are a like-for-like benign comparison; the ROADM
        # runs add classical noise the simulator does not model.
        real_sum["noise_injected"] = ~real_sum["source"].str.startswith("baseline_no_signal")
        losses = np.arange(4.0, 36.0, 2.0)
        sim = simulated_qber_vs_loss(phys, losses)
        wide = sim.pivot(index="loss_db", columns="detector", values="sim_mean_qber").add_prefix("sim_qber_")
        real_sum["nearest_loss"] = real_sum["total_loss_db"].apply(lambda x: losses[np.argmin(abs(losses - x))])
        cal = real_sum.merge(wide, left_on="nearest_loss", right_index=True, how="left")
        cal["data_label"] = REAL_LABEL
        cal.to_csv(METRICS / "real_vs_sim_qber_calibration.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
        for src, g in real_sum.groupby("source"):
            inj = bool(g["noise_injected"].iloc[0])
            ax.errorbar(g["total_loss_db"], g["mean_qber"] * 100, yerr=g["std_qber"] * 100, fmt="o", ms=3,
                        alpha=0.25 if inj else 0.9,
                        label=f"real{' (+ROADM noise)' if inj else ''}: {src.replace('_full_sweep', '').replace('.csv', '')[:30]}")
        for det, g in sim.groupby("detector"):
            ax.fill_between(g["loss_db"], (g["sim_mean_qber"] - g["sim_std_qber"]) * 100,
                            (g["sim_mean_qber"] + g["sim_std_qber"]) * 100, alpha=0.25)
            ax.plot(g["loss_db"], g["sim_mean_qber"] * 100, lw=2, label=f"simulated benign ({det}, ±1 sd)")
        ax.set_xlabel("Total loss (dB)"); ax.set_ylabel("QBER (%)")
        ax.set_title("Real commercial-system QBER vs simulator (benign)", fontsize=11)
        ax.legend(fontsize=6.5, ncol=2); ax.grid(alpha=0.3)
        fig.text(0.01, 0.01, f"Real: Zenodo 21792844 (CC-BY-4.0). Simulated: config/physics.yaml, seed 7.", fontsize=6)
        fig.tight_layout(); fig.savefig(FIGURES / "fig_real_vs_sim_qber.png"); plt.close(fig)
        base = real_sum[~real_sum["noise_injected"]]
        rq = base["mean_qber"]
        print(f"  Real QBER (no noise injection) {rq.min():.4f}-{rq.max():.4f} over {base['total_loss_db'].min():.0f}-"
              f"{base['total_loss_db'].max():.0f} dB ({len(base)} VOA blocks; {int(real_sum['noise_injected'].sum())} "
              f"noise-injected blocks plotted faded)")
        for det, g in sim.groupby("detector"):
            print(f"  Simulated {det:12s} QBER {g['sim_mean_qber'].min():.4f}-{g['sim_mean_qber'].max():.4f} over 4-34 dB")

    # ---- 2. OOD false-alarm test with order-free features ----
    blocks = load_blocks(raw)
    train_path = ROOT_DIR / "data" / "processed" / "train_multiattack.csv"
    summary = {"data_label": REAL_LABEL, "citation": CITATION, "features": ORDER_FREE_FEATURES}
    if blocks.empty or not train_path.exists():
        print("  Need real CSV blocks and data/processed/train_multiattack.csv for the OOD test; skipping.")
    else:
        train = pd.read_csv(train_path, usecols=ORDER_FREE_FEATURES + ["block_class"])
        n_cls = int(train["block_class"].max()) + 1
        model = xgb.XGBClassifier(objective="multi:softprob", num_class=n_cls, n_estimators=200, max_depth=5,
                                  learning_rate=0.08, subsample=0.85, colsample_bytree=0.9, tree_method="hist",
                                  random_state=42, n_jobs=-1)
        model.fit(train[ORDER_FREE_FEATURES].values, train["block_class"].values)
        prob = model.predict_proba(blocks[ORDER_FREE_FEATURES].values)
        blocks["pred_class"] = prob.argmax(axis=1)
        blocks["pred_name"] = [CLASS_NAMES[i] for i in blocks["pred_class"]]
        blocks["p_attack"] = 1.0 - prob[:, 0]
        blocks["false_alarm"] = blocks["pred_class"] != 0
        blocks["fixed11_alarm"] = blocks["qber_mean"] > float(phys["asymptotic_abort_qber"])
        blocks["data_label"] = REAL_LABEL
        blocks.to_csv(METRICS / "real_data_ood_false_alarms.csv", index=False)
        summary.update({
            "real_blocks": int(len(blocks)),
            "model": "XGBoost trained on simulated train split, order-free QBER features only",
            "false_alarm_rate": float(blocks["false_alarm"].mean()),
            "fixed11_false_alarm_rate": float(blocks["fixed11_alarm"].mean()),
            "predicted_class_counts": blocks["pred_name"].value_counts().to_dict(),
            "caveat": ("Real records are sorted by QBER within each attenuation setting and have no "
                       "timestamps, so temporal features cannot be computed; a real record's QBER is "
                       "estimated over a much larger key block than a simulated block."),
        })
        print(f"  {len(blocks)} real blocks; model false-alarm rate {summary['false_alarm_rate']:.3f}; "
              f"fixed-11% false-alarm rate {summary['fixed11_false_alarm_rate']:.3f}")
        print(f"  predicted classes: {summary['predicted_class_counts']}")
    with open(METRICS / "real_data_ood_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
