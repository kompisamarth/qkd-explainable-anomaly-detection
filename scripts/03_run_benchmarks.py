"""Benchmark execution script for BB84 QKD Eavesdropping Detection.

Trains baseline & ML models, calibrates FPR thresholds, and evaluates:
1. Classification accuracy, F1, FPR, and AUROC on Fiber channel
2. Operational Detection Latency Delta_T across fractional interception rates
3. Cross-channel noise generalization degradation on Free-Space atmospheric turbulence
4. Multi-class classification (Benign vs. Full IR vs. Fractional IR)
"""

import sys
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.features.time_series import FEATURE_NAMES, CausalTimeSeriesExtractor
from src.simulator.bb84_engine import BB84Simulator
from src.models.hoeffding_baseline import HoeffdingDetector
from src.models.ml_classifiers import QKDMLDetector
from src.evaluation.benchmarks import (
    evaluate_classification_metrics,
    evaluate_detection_latency,
    compute_cross_channel_matrix,
)


def load_config() -> dict:
    with open(ROOT_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


class StaticThresholdDetector:
    """Standard static QBER abort threshold (11%)."""
    def __init__(self, threshold: float = 0.11):
        self.threshold = threshold

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        qber = X["qber_current"].values if "qber_current" in X.columns else X.iloc[:, 0].values
        return (qber >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        qber = X["qber_current"].values if "qber_current" in X.columns else X.iloc[:, 0].values
        p_attack = np.clip(qber / self.threshold, 0.0, 1.0)
        return np.column_stack([1.0 - p_attack, p_attack])


def main():
    print("=" * 70)
    print("BB84 EXPLAINABLE TEMPORAL ML BENCHMARK SUITE")
    print("=" * 70)

    cfg = load_config()
    proc_dir = ROOT_DIR / cfg["paths"]["processed_data"]
    metrics_dir = ROOT_DIR / cfg["paths"]["metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data Splits
    print("\n[1/5] Loading Processed Datasets...")
    train_df = pd.read_csv(proc_dir / "train_fiber.csv")
    val_df = pd.read_csv(proc_dir / "val_fiber.csv")
    test_fiber_df = pd.read_csv(proc_dir / "test_fiber.csv")
    test_fso_df = pd.read_csv(proc_dir / "test_freespace.csv")

    X_train = train_df[FEATURE_NAMES]
    y_train_bin = train_df["is_attack_active"].values
    y_train_multi = train_df["block_class"].values

    X_val = val_df[FEATURE_NAMES]
    y_val_bin = val_df["is_attack_active"].values

    X_test_fiber = test_fiber_df[FEATURE_NAMES]
    y_test_fiber_bin = test_fiber_df["is_attack_active"].values
    y_test_fiber_multi = test_fiber_df["block_class"].values

    print(f"  Training samples:   {len(train_df):,} blocks")
    print(f"  Validation samples: {len(val_df):,} blocks")
    print(f"  Fiber Test samples: {len(test_fiber_df):,} blocks")
    print(f"  FSO Test samples:   {len(test_fso_df):,} blocks")

    # 2. Instantiate and Train Models
    print("\n[2/5] Training Models and Calibrating Decision Thresholds...")
    detectors = {
        "Static 11% Threshold": StaticThresholdDetector(threshold=0.11),
        "Hoeffding (Single Block)": HoeffdingDetector(base_qber=0.03, alpha=0.01, mode="single_block"),
        "Hoeffding (Windowed W=10)": HoeffdingDetector(base_qber=0.03, alpha=0.01, mode="windowed", window_size=10),
    }

    # Binary ML Detectors
    rf_binary = QKDMLDetector(
        model_type="random_forest",
        mode="binary",
        target_fpr=0.01,
        model_params=cfg["models"]["random_forest"],
    )
    rf_binary.fit(X_train, y_train_bin, X_val, y_val_bin)
    detectors["Random Forest (Calibrated)"] = rf_binary

    xgb_binary = QKDMLDetector(
        model_type="xgboost",
        mode="binary",
        target_fpr=0.01,
        model_params=cfg["models"]["xgboost"],
    )
    xgb_binary.fit(X_train, y_train_bin, X_val, y_val_bin)
    detectors["XGBoost (Calibrated)"] = xgb_binary

    print(f"  RF Calibrated Threshold (FPR <= 1%):  {rf_binary.calibrated_threshold:.4f}")
    print(f"  XGB Calibrated Threshold (FPR <= 1%): {xgb_binary.calibrated_threshold:.4f}")

    # Multi-Class ML Detectors
    rf_multi = QKDMLDetector(
        model_type="random_forest",
        mode="multiclass",
        model_params=cfg["models"]["random_forest"],
    )
    rf_multi.fit(X_train, y_train_multi)

    xgb_multi = QKDMLDetector(
        model_type="xgboost",
        mode="multiclass",
        model_params=cfg["models"]["xgboost"],
    )
    xgb_multi.fit(X_train, y_train_multi)

    # Save model artifacts for XAI
    models_dir = proc_dir / "checkpoints"
    models_dir.mkdir(parents=True, exist_ok=True)
    rf_binary.save(models_dir / "rf_binary.joblib")
    xgb_binary.save(models_dir / "xgb_binary.joblib")
    rf_multi.save(models_dir / "rf_multi.joblib")
    xgb_multi.save(models_dir / "xgb_multi.joblib")

    # 3. Evaluate Binary Classification on Fiber Test Set
    print("\n[3/5] Evaluating Classification Benchmarks on Fiber Test Set...")
    fiber_results = []
    for name, det in detectors.items():
        preds = det.predict(test_fiber_df)
        scores = det.predict_proba(test_fiber_df)[:, 1] if hasattr(det, "predict_proba") else None
        m = evaluate_classification_metrics(y_test_fiber_bin, preds, scores)
        m["model"] = name
        fiber_results.append(m)
        print(f"  {name:30s} | Acc: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | FPR: {m['false_positive_rate']:.4f} | AUROC: {m['auroc']:.4f}")

    df_fiber_benchmarks = pd.DataFrame(fiber_results)
    df_fiber_benchmarks.to_csv(metrics_dir / "classification_benchmarks_fiber.csv", index=False)
    with open(metrics_dir / "classification_benchmarks_fiber.json", "w") as f:
        json.dump(fiber_results, f, indent=2)

    # 4. Operational Detection Latency Benchmarks
    print("\n[4/5] Evaluating Operational Detection Latency Delta_T across p_eve grid...")
    p_eve_grid = cfg["benchmarks"]["latency_p_eve_grid"]
    sim = BB84Simulator(
        raw_photons_per_block=cfg["simulation"]["raw_photons_per_block"],
        base_qber=cfg["physics"]["baseline_qber"],
        random_state=cfg["simulation"]["random_seed"],
    )
    extractor = CausalTimeSeriesExtractor(
        window_size=cfg["features"]["window_size"],
        ema_span=cfg["features"]["ema_span"],
    )

    latency_records = []
    eval_models = {
        "Static 11%": "static_11",
        "Hoeffding Single": detectors["Hoeffding (Single Block)"],
        "Hoeffding Windowed": detectors["Hoeffding (Windowed W=10)"],
        "Random Forest": rf_binary,
        "XGBoost": xgb_binary,
    }

    for m_label, det_obj in eval_models.items():
        print(f"  Measuring latency for: {m_label}...")
        df_lat = evaluate_detection_latency(
            sim=sim,
            extractor=extractor,
            detector=det_obj,
            p_eve_grid=p_eve_grid,
            num_sessions_per_p=40,
            channel_type="fiber",
            blocks_per_session=cfg["simulation"]["blocks_per_session"],
            attack_start_block=cfg["simulation"]["attack_start_block"],
        )
        df_lat["model"] = m_label
        latency_records.append(df_lat)

    df_latency_combined = pd.concat(latency_records, ignore_index=True)
    df_latency_combined.to_csv(metrics_dir / "detection_latency_comparison.csv", index=False)

    # 5. Cross-Channel Noise Generalization Matrix
    print("\n[5/5] Evaluating Cross-Channel Noise Generalization (Fiber vs. Free-Space)...")
    cross_df = compute_cross_channel_matrix(
        models_dict=detectors,
        test_fiber_df=test_fiber_df,
        test_fso_df=test_fso_df,
        feature_names=FEATURE_NAMES,
    )
    cross_df.to_csv(metrics_dir / "cross_channel_generalization.csv", index=False)
    for _, row in cross_df.iterrows():
        print(f"  {row['model']:30s} | Fiber F1: {row['fiber_f1']:.4f} -> FSO F1: {row['fso_f1']:.4f} (Drop: {row['f1_drop']:+.4f}) | FSO FPR: {row['fso_fpr']:.4f}")

    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETED AND METRICS EXPORTED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
