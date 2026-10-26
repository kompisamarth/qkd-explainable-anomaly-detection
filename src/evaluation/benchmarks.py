"""Evaluation and Benchmarking Suite for BB84 Eavesdropping Detection.

Implements:
1. Operational Detection Latency: Delta_T = T_detect - T_attack
2. Standard Classification Metrics (Accuracy, Precision, Recall, F1, AUROC, FPR)
3. Multi-Class Confusion Matrices
4. Cross-Channel Generalization Matrix (Fiber vs. Free-Space Scintillation)
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)

from src.simulator.bb84_engine import BB84Simulator
from src.features.time_series import CausalTimeSeriesExtractor
from src.models.hoeffding_baseline import HoeffdingDetector
from src.models.ml_classifiers import QKDMLDetector


def evaluate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Computes operational classification metrics including exact False Positive Rate."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix elements for exact FPR = FP / (FP + TN)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    if y_scores is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_scores))
            metrics["auprc"] = float(average_precision_score(y_true, y_scores))
        except Exception:
            metrics["auroc"] = 0.50
            metrics["auprc"] = 0.0
    else:
        metrics["auroc"] = 0.50
        metrics["auprc"] = 0.0

    return metrics


def evaluate_detection_latency(
    sim: BB84Simulator,
    extractor: CausalTimeSeriesExtractor,
    detector: Union[HoeffdingDetector, QKDMLDetector, str],
    p_eve_grid: List[float],
    num_sessions_per_p: int = 50,
    channel_type: str = "fiber",
    blocks_per_session: int = 200,
    attack_start_block: int = 101,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Measures empirical detection latency Delta_T = T_detect - T_attack.
    
    For each session:
    - Blocks 1 .. (T_attack - 1) are Benign
    - At T_attack, Eve begins intercepting with probability p_eve
    - T_detect is the first block t >= T_attack where the detector flags an alarm
    - Latency Delta_T = T_detect - T_attack (in units of sifted photon blocks)
    - If no alarm occurs before block B, Delta_T is censored at (B - T_attack + 1)
    """
    rng = np.random.default_rng(random_seed)
    records = []

    for p_eve in p_eve_grid:
        latencies = []
        detected_flags = []
        false_alarms = 0
        total_pre_blocks = 0

        target_class = 1 if p_eve >= 0.99 else (2 if p_eve > 0 else 0)

        for s in range(num_sessions_per_p):
            sess_id = 90000 + int(p_eve * 1000) + s
            seq = sim.simulate_session(
                session_id=sess_id,
                target_attack_class=target_class,
                channel_type=channel_type,
                blocks_per_session=blocks_per_session,
                attack_start_block=attack_start_block,
                p_eve_fractional=p_eve,
            )

            df_feat = extractor.extract_from_session(
                session_id=sess_id,
                qbers=seq.block_qbers,
                sifted_lens=seq.block_sifted_lens,
                block_classes=seq.block_classes,
                attack_class=seq.attack_class,
                channel_type=seq.channel_type,
                p_eve=p_eve,
                attack_start_block=attack_start_block,
            )

            # Generate predictions
            if isinstance(detector, str) and detector == "static_11":
                preds = (df_feat["qber_current"] >= 0.11).astype(int).values
            else:
                preds = detector.predict(df_feat)
                if hasattr(preds, "values"):
                    preds = preds.values

            block_indices = df_feat["block_index"].values

            # Pre-attack blocks check for false alarm rate
            pre_mask = (block_indices < attack_start_block)
            false_alarms += int(np.sum(preds[pre_mask]))
            total_pre_blocks += int(np.sum(pre_mask))

            # Post-attack blocks check for first detection
            post_mask = (block_indices >= attack_start_block)
            post_indices = block_indices[post_mask]
            post_preds = preds[post_mask]

            alarm_indices = post_indices[post_preds == 1]
            if len(alarm_indices) > 0:
                t_detect = alarm_indices[0]
                delta_t = int(t_detect - attack_start_block)
                latencies.append(delta_t)
                detected_flags.append(1)
            else:
                # Censored latency: maximum remaining blocks in session
                latencies.append(blocks_per_session - attack_start_block + 1)
                detected_flags.append(0)

        records.append({
            "p_eve": p_eve,
            "mean_delta_t": float(np.mean(latencies)),
            "median_delta_t": float(np.median(latencies)),
            "detection_rate": float(np.mean(detected_flags)),
            "pre_attack_fpr": float(false_alarms / total_pre_blocks) if total_pre_blocks > 0 else 0.0,
            "num_sessions": num_sessions_per_p,
        })

    return pd.DataFrame(records)


def compute_cross_channel_matrix(
    models_dict: Dict[str, Any],
    test_fiber_df: pd.DataFrame,
    test_fso_df: pd.DataFrame,
    feature_names: List[str]
) -> pd.DataFrame:
    """Evaluates trained models on in-distribution Fiber and out-of-distribution Free-Space data."""
    results = []

    y_fiber = test_fiber_df["is_attack_active"].values
    y_fso = test_fso_df["is_attack_active"].values

    for model_name, model in models_dict.items():
        # Evaluate on Fiber
        pred_fiber = model.predict(test_fiber_df)
        scores_fiber = model.predict_proba(test_fiber_df)[:, 1] if hasattr(model, "predict_proba") else None
        m_fiber = evaluate_classification_metrics(y_fiber, pred_fiber, scores_fiber)

        # Evaluate on Free-Space Scintillation
        pred_fso = model.predict(test_fso_df)
        scores_fso = model.predict_proba(test_fso_df)[:, 1] if hasattr(model, "predict_proba") else None
        m_fso = evaluate_classification_metrics(y_fso, pred_fso, scores_fso)

        results.append({
            "model": model_name,
            "fiber_accuracy": m_fiber["accuracy"],
            "fiber_f1": m_fiber["f1_score"],
            "fiber_auroc": m_fiber["auroc"],
            "fiber_fpr": m_fiber["false_positive_rate"],
            "fso_accuracy": m_fso["accuracy"],
            "fso_f1": m_fso["f1_score"],
            "fso_auroc": m_fso["auroc"],
            "fso_fpr": m_fso["false_positive_rate"],
            "f1_drop": m_fiber["f1_score"] - m_fso["f1_score"],
            "fpr_increase": m_fso["false_positive_rate"] - m_fiber["false_positive_rate"],
        })

    return pd.DataFrame(results)
