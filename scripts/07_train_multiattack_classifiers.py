"""Phase 2: Train and tune multi-class XGBoost classifier for 5 operational classes.

Classes:
- 0: Normal Channel
- 1: Fractional Intercept-Resend (IR)
- 2: Photon Number Splitting (PNS)
- 3: Fake State (Detector Blinding)
- 4: Time-Shift Attack

Performs 5-fold stratified cross-validation on the 63 physics-informed features,
trains the final model, and persists the checkpoint.
"""

import sys
import json
import time
from pathlib import Path
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, log_loss
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.features.time_series import FEATURE_NAMES_63


def load_config() -> dict:
    with open(ROOT_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    ma_cfg = cfg.get("multiattack", {})
    xgb_cfg = ma_cfg.get("xgboost", {})
    seed = ma_cfg.get("random_seed", 42)

    data_dir = ROOT_DIR / "data" / "processed"
    train_path = data_dir / "train_multiattack.csv"
    test_path = data_dir / "test_multiattack.csv"

    print("=" * 70)
    print("PHASE 2: TRAINING MULTI-CLASS XGBOOST CLASSIFIER (5 CLASSES)")
    print(f"  Train path: {train_path}")
    print(f"  Test path:  {test_path}")
    print(f"  Features:   {len(FEATURE_NAMES_63)} physics-informed features")
    print("=" * 70)

    print("\nLoading dataset...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[FEATURE_NAMES_63].values
    y_train = train_df["block_class"].values.astype(int)

    X_test = test_df[FEATURE_NAMES_63].values
    y_test = test_df["block_class"].values.astype(int)

    print(f"  Training shape:  {X_train.shape} across 5 classes")
    print(f"  Test shape:      {X_test.shape}")

    # -------------------------------------------------------------
    # 1. 5-Fold Stratified Cross-Validation on Train Set
    # -------------------------------------------------------------
    print("\n[1/3] Running 5-Fold Stratified Cross-Validation on Training Partition...")
    # Subsample for CV if dataset is huge for rapid tuning
    cv_subsample_idx = np.random.default_rng(seed).choice(len(X_train), size=min(len(X_train), 100000), replace=False)
    X_cv = X_train[cv_subsample_idx]
    y_cv = y_train[cv_subsample_idx]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_accs, cv_f1s, cv_losses = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv), 1):
        fold_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            eval_metric="mlogloss",
            tree_method="hist",
            n_estimators=150,
            max_depth=xgb_cfg.get("max_depth", 6),
            learning_rate=xgb_cfg.get("learning_rate", 0.08),
            subsample=xgb_cfg.get("subsample", 0.85),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.80),
            random_state=seed + fold,
            n_jobs=-1,
        )
        fold_model.fit(X_cv[train_idx], y_cv[train_idx])
        probs = fold_model.predict_proba(X_cv[val_idx])
        preds = np.argmax(probs, axis=1)

        acc = accuracy_score(y_cv[val_idx], preds)
        f1 = f1_score(y_cv[val_idx], preds, average="macro")
        loss = log_loss(y_cv[val_idx], probs)

        cv_accs.append(acc)
        cv_f1s.append(f1)
        cv_losses.append(loss)
        print(f"  Fold {fold}: Acc = {acc:.4f} | Macro-F1 = {f1:.4f} | LogLoss = {loss:.4f}")

    print(f"\n  Mean CV Accuracy:  {np.mean(cv_accs):.4f} (+/- {np.std(cv_accs):.4f})")
    print(f"  Mean CV Macro-F1:  {np.mean(cv_f1s):.4f} (+/- {np.std(cv_f1s):.4f})")
    print(f"  Mean CV LogLoss:   {np.mean(cv_losses):.4f}")

    # -------------------------------------------------------------
    # 2. Train Final Model on Complete Training Set
    # -------------------------------------------------------------
    print("\n[2/3] Training Final Model on Full Training Set...")
    t0 = time.time()
    final_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=xgb_cfg.get("n_estimators", 250),
        max_depth=xgb_cfg.get("max_depth", 6),
        learning_rate=xgb_cfg.get("learning_rate", 0.08),
        subsample=xgb_cfg.get("subsample", 0.85),
        colsample_bytree=xgb_cfg.get("colsample_bytree", 0.80),
        random_state=seed,
        n_jobs=-1,
    )
    final_model.fit(X_train, y_train)
    t1 = time.time()
    print(f"  Model training finished in {t1 - t0:.2f} seconds.")

    # -------------------------------------------------------------
    # 3. Model Checkpointing & Metric Logging
    # -------------------------------------------------------------
    print("\n[3/3] Saving Checkpoint and Metrics...")
    ckpt_dir = data_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "xgb_multiattack.joblib"
    joblib.dump(final_model, ckpt_path)
    print(f"  Saved trained model to: {ckpt_path}")

    metrics_dir = ROOT_DIR / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cv_summary = {
        "model": "XGBClassifier (multi:softprob, 5 classes)",
        "num_features": len(FEATURE_NAMES_63),
        "cv_folds": 5,
        "mean_accuracy": float(np.mean(cv_accs)),
        "std_accuracy": float(np.std(cv_accs)),
        "mean_macro_f1": float(np.mean(cv_f1s)),
        "std_macro_f1": float(np.std(cv_f1s)),
        "mean_logloss": float(np.mean(cv_losses)),
        "train_samples": len(X_train),
        "training_time_sec": float(t1 - t0),
    }
    with open(metrics_dir / "cv_multiattack_xgboost.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    print("=" * 70)
    print("XGBOOST MODEL TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
