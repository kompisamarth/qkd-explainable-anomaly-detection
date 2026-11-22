"""Generate the multi-attack dataset (session-split train/test + held-out generalisation set).

Engine "calibrated" (default, config/physics.yaml):
  * 7 classes: Normal, intercept-resend, PNS, detector blinding, time-shift,
    Trojan horse (simplified), beam splitting.
  * Channel, detector and noise drawn from ONE distribution for every class.
  * train / test split by session id (stratified by class); no session in both.
  * heldout_generalization: attack intensities and fibre distances outside the
    training ranges, saved separately.
  * CSV + Parquet + <name>.meta.json sidecars; leakage audit written to
    results/metrics/multiattack_dataset_leakage_audit.json.

Engine "legacy" reproduces the original 5-class P2.1 generator (bb84_engine).

    python scripts/06_generate_multiattack_dataset.py            # full size from config
    python scripts/06_generate_multiattack_dataset.py --smoke    # tiny run for CI
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.features.time_series import CausalMultiAttackFeatureExtractor, FEATURE_NAMES_63


def load_config() -> dict:
    """config.yaml merged with config/physics.yaml (see src/config.py)."""
    from src.config import load_config as _load
    return _load()


def generate_legacy(ma_cfg, out_dir):
    """Original P2.1 generator, unchanged apart from the shared config loader."""
    from src.simulator.bb84_engine import BB84Simulator
    seed = ma_cfg.get("random_seed", 42)
    spc = ma_cfg.get("sessions_per_class", 1000)
    sim = BB84Simulator(raw_photons_per_block=ma_cfg.get("raw_photons_per_block", 1024), random_state=seed)
    ex = CausalMultiAttackFeatureExtractor(window_size=ma_cfg.get("window_size", 50), ema_span=ma_cfg.get("ema_span", 10))
    train, test, sid = [], [], 0
    for cls in tqdm(range(5), desc="classes"):
        for i in range(spc):
            sid += 1
            seq = sim.simulate_session(session_id=sid, target_attack_class=cls, channel_type="fiber",
                                       blocks_per_session=ma_cfg.get("blocks_per_session", 150),
                                       attack_start_block=ma_cfg.get("attack_start_block", 76),
                                       attack_mode="multiattack", randomize_nuisance=True)
            (train if i < int(spc * ma_cfg.get("train_split", 0.8)) else test).append(ex.extract_from_session(seq))
    pd.concat(train, ignore_index=True).to_csv(out_dir / "train_multiattack.csv", index=False)
    pd.concat(test, ignore_index=True).to_csv(out_dir / "test_multiattack.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny dataset for CI (8 sessions/class)")
    ap.add_argument("--sessions-per-class", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config()
    ma = cfg["multiattack"]
    out_dir = Path(args.out_dir) if args.out_dir else ROOT_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if ma.get("engine", "calibrated") == "legacy":
        generate_legacy(ma, out_dir)
        return

    from src.data.dataset import export, generate_sessions, leakage_audit, model_features, session_split
    from src.simulator.calibrated import CLASS_NAMES

    phys = cfg["calibrated_bb84"]
    seed = int(ma.get("random_seed", 42))
    classes = list(ma.get("class_ids", range(len(CLASS_NAMES))))
    spc = args.sessions_per_class or (8 if args.smoke else int(ma["sessions_per_class"]))
    held_spc = 3 if args.smoke else int(ma.get("heldout_sessions_per_class", 100))
    common = dict(classes=classes, channel=ma.get("channel", "fiber"), blocks=int(ma["blocks_per_session"]),
                  onset=int(ma["attack_start_block"]), window=int(ma["window_size"]), ema=int(ma["ema_span"]),
                  physics=phys)

    print("=" * 74)
    print(f"CALIBRATED MULTI-ATTACK DATASET  ({len(classes)} classes, {spc} sessions/class, seed {seed})")
    print("=" * 74)
    t0 = time.time()
    pool, pool_meta = generate_sessions(spc, seed, heldout=False, first_session_id=1, **common)
    train, test = session_split(pool, float(ma.get("test_split", 0.2)), seed)
    held, held_meta = generate_sessions(held_spc, seed + 1_000_003, heldout=True,
                                        first_session_id=10_000_000, **common)
    print(f"  generated in {time.time() - t0:.0f}s: pool {len(pool):,} rows, held-out {len(held):,} rows")
    assert not set(train["session_id"]) & set(test["session_id"]), "session in both train and test"

    feats = model_features(train)
    audit = leakage_audit(train, feats, len(CLASS_NAMES), seed=seed)
    for k, v in audit.items():
        if isinstance(v, dict):
            flag = "  <-- POSSIBLE LEAKAGE" if v["flagged"] else ""
            print(f"  leakage audit class {k} {v['class_name']:32s} best AUC {v['best_auc']:.3f} via {v['feature']}{flag}")

    att = phys["attacks"]
    ranges = {
        "training": {
            "fiber_distance_km": phys["channel"]["fiber"]["distance_km"],
            "attack_intensity": {a: att[a]["intensity"] for a in att},
        },
        "heldout": {
            "fiber_distance_km": phys["heldout"]["distance_km"],
            "attack_intensity": {a: att[a]["heldout_intensity"] for a in att},
        },
        "detector_types": phys["detector"]["types"],
        "decoy_enabled": phys["source"]["decoy"]["enabled"],
    }
    base_meta = {"seed": seed, "classes": {str(c): CLASS_NAMES[c] for c in classes}, "features": feats,
                 "parameter_ranges": ranges, "engine": "calibrated", "smoke": bool(args.smoke)}
    suffix = "_smoke" if args.smoke else ""
    export(train, out_dir / f"train_multiattack{suffix}", {**base_meta, "split": "train (session ids disjoint from test)"})
    export(test, out_dir / f"test_multiattack{suffix}", {**base_meta, "split": "test (same distributions, unseen sessions)"})
    export(held, out_dir / f"heldout_generalization{suffix}",
           {**base_meta, "seed": seed + 1_000_003, "split": "held-out intensities and distances"})
    pd.concat([pool_meta.assign(split=np.where(pool_meta["session_id"].isin(test["session_id"]), "test", "train")),
               held_meta.assign(split="heldout")]).to_csv(out_dir / f"sessions_multiattack{suffix}.csv", index=False)

    if not args.smoke:
        metrics = ROOT_DIR / "results" / "metrics"
        metrics.mkdir(parents=True, exist_ok=True)
        with open(metrics / "multiattack_dataset_leakage_audit.json", "w", encoding="utf-8") as f:
            json.dump({"alert_threshold_auc": 0.95, "computed_on": "train split", "seed": seed,
                       "classes": audit, "parameter_ranges": ranges}, f, indent=2)
    print(f"  wrote {out_dir}/train|test_multiattack{suffix}, heldout_generalization{suffix} (+ .parquet, .meta.json)")


if __name__ == "__main__":
    main()
