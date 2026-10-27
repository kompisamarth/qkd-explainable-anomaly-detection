"""Machine Learning Classifiers (Random Forest and XGBoost) for BB84 QKD.

Includes support for:
- Binary anomaly detection (Benign vs. Attack)
- Multi-class classification (Class 0: Benign, Class 1: Full IR, Class 2: Fractional IR)
- Empirical threshold calibration to match Hoeffding False Positive Rate constraint (FPR <= 1%)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib


class QKDMLDetector:
    """Wrapper for Random Forest or XGBoost with threshold calibration for FPR control."""

    def __init__(
        self,
        model_type: str = "random_forest",
        mode: str = "binary",  # 'binary' or 'multiclass'
        target_fpr: float = 0.01,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        self.model_type = model_type.lower()
        self.mode = mode.lower()
        self.target_fpr = float(target_fpr)
        self.calibrated_threshold = 0.50
        self.feature_names: List[str] = []

        params = model_params or {}
        if self.model_type == "random_forest":
            rf_defaults = {
                "n_estimators": 150,
                "max_depth": 8,
                "random_state": 42,
                "n_jobs": -1,
                "class_weight": "balanced" if self.mode == "binary" else None,
            }
            rf_defaults.update(params)
            self.model = RandomForestClassifier(**rf_defaults)
        elif self.model_type == "xgboost":
            xgb_defaults = {
                "n_estimators": 150,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "eval_metric": "logloss" if self.mode == "binary" else "mlogloss",
            }
            xgb_defaults.update(params)
            self.model = XGBClassifier(**xgb_defaults)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "QKDMLDetector":
        """Fits the underlying model and calibrates decision threshold on validation set."""
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)

        # Calibrate decision threshold for binary classification
        if self.mode == "binary" and X_val is not None and y_val is not None:
            self.calibrate_threshold(X_val, y_val)
        return self

    def calibrate_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: Union[pd.Series, np.ndarray]
    ) -> float:
        """Finds the decision threshold tau that keeps FPR <= target_fpr on benign samples."""
        y_val_arr = np.asarray(y_val)
        probs = self.model.predict_proba(X_val)[:, 1]
        
        benign_mask = (y_val_arr == 0)
        benign_probs = probs[benign_mask]

        if len(benign_probs) == 0:
            self.calibrated_threshold = 0.50
            return 0.50

        # Target 99th percentile of benign probabilities (or 1 - target_fpr)
        quantile_target = 100.0 * (1.0 - self.target_fpr)
        threshold_candidate = float(np.percentile(benign_probs, quantile_target))
        # Ensure threshold is at least 0.50 and at most 0.99
        self.calibrated_threshold = float(np.clip(threshold_candidate, 0.50, 0.99))
        return self.calibrated_threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns class probabilities."""
        return self.model.predict_proba(X[self.feature_names])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predictions. For binary mode, uses the FPR-calibrated threshold."""
        X_sub = X[self.feature_names]
        if self.mode == "binary":
            probs = self.model.predict_proba(X_sub)[:, 1]
            return (probs >= self.calibrated_threshold).astype(int)
        else:
            return self.model.predict(X_sub)

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves model and calibration parameters."""
        state = {
            "model_type": self.model_type,
            "mode": self.mode,
            "target_fpr": self.target_fpr,
            "calibrated_threshold": self.calibrated_threshold,
            "feature_names": self.feature_names,
            "model": self.model,
        }
        joblib.dump(state, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "QKDMLDetector":
        """Loads saved model and configuration."""
        state = joblib.load(filepath)
        detector = cls(
            model_type=state["model_type"],
            mode=state["mode"],
            target_fpr=state["target_fpr"],
        )
        detector.calibrated_threshold = state["calibrated_threshold"]
        detector.feature_names = state["feature_names"]
        detector.model = state["model"]
        return detector
