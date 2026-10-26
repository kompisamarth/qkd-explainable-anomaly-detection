"""Multi-condition synthetic dataset generator for BB84 QKD anomaly detection.

Generates chronological transmission sessions across Fiber and Free-Space channels,
applies causal sliding-window feature engineering, and exports labeled splits.
"""

import sys
from pathlib import Path
from typing import Tuple
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.simulator.bb84_engine import BB84Simulator
from src.features.time_series import CausalTimeSeriesExtractor


def load_config() -> dict:
    config_path = ROOT_DIR / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_split_sessions(
    sim: BB84Simulator,
    extractor: CausalTimeSeriesExtractor,
    num_sessions: int,
    channel_type: str,
    start_session_id: int,
    cfg: dict,
    desc: str = "Generating"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates balanced sessions across Class 0, 1, and 2, and extracts causal features."""
    blocks_per_session = cfg["simulation"]["blocks_per_session"]
    attack_start = cfg["simulation"]["attack_start_block"]
    fiber_qber = cfg["physics"]["baseline_qber"]
    scint_std = cfg["physics"]["scintillation_std"]
    ar1_phi = cfg["physics"]["scintillation_ar1_phi"]

    # Balanced allocation across Class 0, 1, 2
    classes = [0, 1, 2] * (num_sessions // 3)
    if len(classes) < num_sessions:
        classes += [0, 1, 2][: (num_sessions - len(classes))]

    feature_dfs = []
    session_metadata = []

    for i, target_class in enumerate(tqdm(classes, desc=desc)):
        sess_id = start_session_id + i
        session_seq = sim.simulate_session(
            session_id=sess_id,
            target_attack_class=target_class,
            channel_type=channel_type,
            blocks_per_session=blocks_per_session,
            attack_start_block=attack_start,
            fiber_base_qber=fiber_qber,
            scintillation_std=scint_std,
            ar1_phi=ar1_phi,
        )

        # Extract features
        df_feat = extractor.extract_from_session(
            session_id=sess_id,
            qbers=session_seq.block_qbers,
            sifted_lens=session_seq.block_sifted_lens,
            block_classes=session_seq.block_classes,
            attack_class=session_seq.attack_class,
            channel_type=session_seq.channel_type,
            p_eve=session_seq.p_eve,
            attack_start_block=session_seq.attack_start_block,
        )
        feature_dfs.append(df_feat)

        session_metadata.append({
            "session_id": sess_id,
            "channel_type": channel_type,
            "target_class": target_class,
            "p_eve": session_seq.p_eve,
            "mean_qber": float(np.mean(session_seq.block_qbers)),
            "max_qber": float(np.max(session_seq.block_qbers)),
            "attack_start_block": attack_start,
            "num_blocks": blocks_per_session,
        })

    combined_features = pd.concat(feature_dfs, ignore_index=True)
    combined_meta = pd.DataFrame(session_metadata)
    return combined_features, combined_meta


def main():
    print("=" * 70)
    print("BB84 MULTI-CONDITION DATASET GENERATION PIPELINE")
    print("=" * 70)

    cfg = load_config()
    raw_photons = cfg["simulation"]["raw_photons_per_block"]
    base_qber = cfg["physics"]["baseline_qber"]
    seed = cfg["simulation"]["random_seed"]
    window_size = cfg["features"]["window_size"]
    ema_span = cfg["features"]["ema_span"]

    sim = BB84Simulator(raw_photons_per_block=raw_photons, base_qber=base_qber, random_state=seed)
    extractor = CausalTimeSeriesExtractor(window_size=window_size, ema_span=ema_span)

    n_train = cfg["benchmarks"]["num_sessions_train"]
    n_val = cfg["benchmarks"]["num_sessions_val"]
    n_test = cfg["benchmarks"]["num_sessions_test"]

    raw_dir = ROOT_DIR / cfg["paths"]["raw_data"]
    proc_dir = ROOT_DIR / cfg["paths"]["processed_data"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train Fiber
    print(f"\n[1/4] Generating Fiber Training Set ({n_train} sessions)...")
    train_feat, train_meta = generate_split_sessions(
        sim, extractor, n_train, channel_type="fiber", start_session_id=1000, cfg=cfg, desc="Train Fiber"
    )
    train_path = proc_dir / "train_fiber.csv"
    train_feat.to_csv(train_path, index=False)
    print(f"  Saved {len(train_feat):,} feature rows to {train_path.name}")

    # 2. Val Fiber
    print(f"\n[2/4] Generating Fiber Validation Set ({n_val} sessions)...")
    val_feat, val_meta = generate_split_sessions(
        sim, extractor, n_val, channel_type="fiber", start_session_id=3000, cfg=cfg, desc="Val Fiber"
    )
    val_path = proc_dir / "val_fiber.csv"
    val_feat.to_csv(val_path, index=False)
    print(f"  Saved {len(val_feat):,} feature rows to {val_path.name}")

    # 3. Test Fiber
    print(f"\n[3/4] Generating Fiber Test Set ({n_test} sessions)...")
    test_feat, test_meta = generate_split_sessions(
        sim, extractor, n_test, channel_type="fiber", start_session_id=5000, cfg=cfg, desc="Test Fiber"
    )
    test_path = proc_dir / "test_fiber.csv"
    test_feat.to_csv(test_path, index=False)
    print(f"  Saved {len(test_feat):,} feature rows to {test_path.name}")

    # 4. Test Free-Space (Atmospheric Scintillation)
    print(f"\n[4/4] Generating Free-Space Non-Stationary Test Set ({n_test} sessions)...")
    fso_feat, fso_meta = generate_split_sessions(
        sim, extractor, n_test, channel_type="free_space", start_session_id=8000, cfg=cfg, desc="Test Free-Space"
    )
    fso_path = proc_dir / "test_freespace.csv"
    fso_feat.to_csv(fso_path, index=False)
    print(f"  Saved {len(fso_feat):,} feature rows to {fso_path.name}")

    # Save metadata summary
    all_meta = pd.concat([train_meta, val_meta, test_meta, fso_meta], ignore_index=True)
    summary_path = raw_dir / "sessions_summary.csv"
    all_meta.to_csv(summary_path, index=False)
    print(f"\nSaved metadata summary for {len(all_meta):,} sessions to {summary_path.name}")

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
