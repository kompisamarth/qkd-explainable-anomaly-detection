"""Phase 2: Generate multi-class synthetic dataset across 5 operational classes.

Classes:
- 0: Normal Channel (Benign)
- 1: Fractional / Stealthy Intercept-Resend (IR)
- 2: Photon Number Splitting (PNS)
- 3: Fake State (Detector Blinding)
- 4: Time-Shift Attack

Extracts 63 physics-informed features over causal sliding windows of length W=50.
Partitions with a stratified 80/20 session-level train/test split.
"""

import sys
import time
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.simulator.bb84_engine import BB84Simulator
from src.features.time_series import CausalMultiAttackFeatureExtractor, FEATURE_NAMES_63


def load_config() -> dict:
    with open(ROOT_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    ma_cfg = cfg.get("multiattack", {})
    seed = ma_cfg.get("random_seed", 42)
    rng = np.random.default_rng(seed)

    sessions_per_class = ma_cfg.get("sessions_per_class", 1000)
    blocks_per_session = ma_cfg.get("blocks_per_session", 150)
    attack_start_block = ma_cfg.get("attack_start_block", 76)
    raw_photons = ma_cfg.get("raw_photons_per_block", 1024)
    w_size = ma_cfg.get("window_size", 50)
    ema_span = ma_cfg.get("ema_span", 10)
    train_split = ma_cfg.get("train_split", 0.80)

    print("=" * 70)
    print("PHASE 2: GENERATING MULTI-ATTACK BB84 DATASET (5 CLASSES, 63 FEATURES)")
    print(f"  Sessions per class:    {sessions_per_class}")
    print(f"  Total sessions:        {sessions_per_class * 5}")
    print(f"  Blocks per session:    {blocks_per_session} (Attack onset at block {attack_start_block})")
    print(f"  Window size W:         {w_size} blocks")
    print(f"  Physics features:      {len(FEATURE_NAMES_63)}")
    print("=" * 70)

    sim = BB84Simulator(raw_photons_per_block=raw_photons, random_state=seed)
    extractor = CausalMultiAttackFeatureExtractor(window_size=w_size, ema_span=ema_span)

    train_dfs = []
    test_dfs = []

    t0 = time.time()
    total_sessions = sessions_per_class * 5

    with tqdm(total=total_sessions, desc="Generating sessions") as pbar:
        session_counter = 0
        for class_id in range(5):
            for s_idx in range(sessions_per_class):
                session_id = session_counter + 1
                session_counter += 1

                # Slight background noise jitter to prevent label leakage
                base_qber = float(np.clip(rng.normal(0.03, 0.003), 0.015, 0.045))

                seq = sim.simulate_session(
                    session_id=session_id,
                    target_attack_class=class_id,
                    channel_type="fiber",
                    blocks_per_session=blocks_per_session,
                    attack_start_block=attack_start_block,
                    fiber_base_qber=base_qber,
                    attack_mode="multiattack",
                )

                df_session = extractor.extract_from_session(seq)

                # 80/20 train/test assignment by session index within class
                if s_idx < int(sessions_per_class * train_split):
                    train_dfs.append(df_session)
                else:
                    test_dfs.append(df_session)

                pbar.update(1)

    print("\nConcatenating processed feature matrices...")
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    out_dir = ROOT_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Saving train and test datasets...")
    # Save CSV
    train_csv = out_dir / "train_multiattack.csv"
    test_csv = out_dir / "test_multiattack.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    t1 = time.time()
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print(f"  Elapsed time:      {t1 - t0:.2f} seconds")
    print(f"  Training samples:  {len(train_df):,} rows -> {train_csv}")
    print(f"  Testing samples:   {len(test_df):,} rows -> {test_csv}")
    print("  Class distribution (block_class in test set):")
    print(test_df["block_class"].value_counts().sort_index().to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
