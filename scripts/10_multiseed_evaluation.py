"""Phase 2 (P2.1): leakage-free, multi-seed evaluation of the 5-class classifier.

Background
----------
The originally committed Phase 2 result (99.93% accuracy) was not a trustworthy
generalization estimate. Every class was simulated at a fixed channel
transmittance of 1.0 and a fixed dark-count rate, so `transmission_efficiency`
collapsed onto a per-class constant -- exactly 1.000 with zero variance for the
Normal and Fractional-IR classes -- and separated three of the five classes on
its own. Each attack also carried hard-coded strength constants (fake-state
forced the residual error to a fixed cap, time-shift hard-coded a 1.0/0.4
basis-efficiency ratio). The model was partly learning the simulator.

What this script does
---------------------
1. Regenerates sessions with `randomize_nuisance=True`, so base QBER, channel
   transmittance, dark counts, per-basis detector efficiency and every attack
   strength parameter are drawn per session from the deliberately OVERLAPPING
   `NUISANCE_RANGES` in src/simulator/bb84_engine.py.
2. Audits single-feature separability (best one-vs-rest AUC per class). If any
   feature still scores near 1.0, a deterministic fingerprint remains.
3. Trains and evaluates XGBoost over N independent seeds and reports
   accuracy / macro-F1 / per-class F1 / FNR as mean +/- std.
4. Compares against the fixed 11% QBER threshold on attack-vs-normal detection.

Train and test never share a session, so overlapping windows from one session
cannot straddle the split.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.features.time_series import FEATURE_NAMES_63, CausalMultiAttackFeatureExtractor
from src.simulator.bb84_engine import NUISANCE_RANGES, BB84Simulator

CLASS_NAMES = ["Normal Channel", "Fractional IR", "PNS Attack", "Fake State", "Time-Shift"]


def load_config() -> dict:
    with open(ROOT_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def generate(seed, sessions_per_class, blocks, attack_start, raw_photons, w, ema, train_split):
    """Session-level 80/20 split; nuisance parameters randomized per session."""
    sim = BB84Simulator(raw_photons_per_block=raw_photons, random_state=seed)
    ex = CausalMultiAttackFeatureExtractor(window_size=w, ema_span=ema)
    train_dfs, test_dfs = [], []
    n_train = int(sessions_per_class * train_split)
    sid = 0
    for cls in range(5):
        for i in range(sessions_per_class):
            sid += 1
            seq = sim.simulate_session(
                session_id=sid,
                target_attack_class=cls,
                channel_type="fiber",
                blocks_per_session=blocks,
                attack_start_block=attack_start,
                attack_mode="multiattack",
                randomize_nuisance=True,
            )
            (train_dfs if i < n_train else test_dfs).append(ex.extract_from_session(seq))
    return pd.concat(train_dfs, ignore_index=True), pd.concat(test_dfs, ignore_index=True)


def leakage_audit(df):
    """Best single-feature one-vs-rest AUC per attack class."""
    y = df["block_class"].values.astype(int)
    out = {}
    for cls in range(1, 5):
        yb = (y == cls).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        best_auc, best_feat = 0.0, ""
        for f in FEATURE_NAMES_63:
            v = df[f].values
            if not np.isfinite(v).all():
                continue
            a = roc_auc_score(yb, v)
            a = max(a, 1.0 - a)
            if a > best_auc:
                best_auc, best_feat = a, f
        out[cls] = {"best_auc": float(best_auc), "feature": best_feat}
    return out


def run_seed(seed, cfg_ma, sessions_per_class, verbose=True):
    t0 = time.time()
    train_df, test_df = generate(
        seed=seed,
        sessions_per_class=sessions_per_class,
        blocks=cfg_ma.get("blocks_per_session", 150),
        attack_start=cfg_ma.get("attack_start_block", 76),
        raw_photons=cfg_ma.get("raw_photons_per_block", 1024),
        w=cfg_ma.get("window_size", 50),
        ema=cfg_ma.get("ema_span", 10),
        train_split=cfg_ma.get("train_split", 0.80),
    )
    t_gen = time.time() - t0

    X_tr = train_df[FEATURE_NAMES_63].values
    y_tr = train_df["block_class"].values.astype(int)
    X_te = test_df[FEATURE_NAMES_63].values
    y_te = test_df["block_class"].values.astype(int)

    xgb_cfg = cfg_ma.get("xgboost", {})
    model = xgb.XGBClassifier(
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
    t1 = time.time()
    model.fit(X_tr, y_tr)
    t_train = time.time() - t1

    pred = model.predict(X_te)

    # ---- 5-class metrics ----
    acc = accuracy_score(y_te, pred)
    macro_f1 = f1_score(y_te, pred, average="macro")
    macro_p = precision_score(y_te, pred, average="macro", zero_division=0)
    macro_r = recall_score(y_te, pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_te, pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)

    # ---- attack-vs-normal comparison ----
    y_bin = (y_te > 0).astype(int)
    xgb_bin = (pred > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, xgb_bin, labels=[0, 1]).ravel()
    xgb_fnr = fn / max(fn + tp, 1)
    xgb_fpr = fp / max(fp + tn, 1)

    # Fixed 11% threshold applied to the window-mean QBER
    thr_bin = (test_df["qber_mean"].values > 0.11).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_bin, thr_bin, labels=[0, 1]).ravel()
    thr_fnr = fn2 / max(fn2 + tp2, 1)
    thr_fpr = fp2 / max(fp2 + tn2, 1)

    audit = leakage_audit(test_df)

    if verbose:
        print(
            f"  seed {seed}: acc={acc:.4f} macroF1={macro_f1:.4f} FNR={xgb_fnr:.4f} "
            f"| gen {t_gen:.0f}s train {t_train:.0f}s | test rows {len(test_df):,}",
            flush=True,
        )

    return {
        "seed": seed,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "xgb_binary_fnr": float(xgb_fnr),
        "xgb_binary_fpr": float(xgb_fpr),
        "fixed11_binary_fnr": float(thr_fnr),
        "fixed11_binary_fpr": float(thr_fpr),
        **{f"f1_class_{i}": float(v) for i, v in enumerate(per_class_f1)},
        "max_single_feature_auc": float(max(a["best_auc"] for a in audit.values())),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "_audit": audit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--sessions-per-class", type=int, default=400)
    args = ap.parse_args()

    cfg_ma = load_config().get("multiattack", {})

    print("=" * 78)
    print("PHASE 2 / P2.1: LEAKAGE-FREE MULTI-SEED EVALUATION")
    print(f"  Seeds:               {args.seeds}")
    print(f"  Sessions per class:  {args.sessions_per_class}  ({args.sessions_per_class * 5} per seed)")
    print(f"  Features:            {len(FEATURE_NAMES_63)}")
    print("  Nuisance ranges (overlapping across all classes):")
    for k, v in NUISANCE_RANGES.items():
        print(f"    {k:26s} {v}")
    print("=" * 78, flush=True)

    rows = []
    t0 = time.time()
    for seed in args.seeds:
        rows.append(run_seed(seed, cfg_ma, args.sessions_per_class))

    audits = [r.pop("_audit") for r in rows]
    df = pd.DataFrame(rows)

    metrics_dir = ROOT_DIR / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_dir / "multiattack_multiseed_per_seed.csv", index=False)

    num = df.drop(columns=["seed"]).select_dtypes("number")
    summary = pd.DataFrame({"mean": num.mean(), "std": num.std(ddof=1)})
    summary.to_csv(metrics_dir / "multiattack_multiseed_summary.csv")

    print("\n" + "=" * 78)
    print(f"RESULTS  (mean +/- std over {len(args.seeds)} seeds)")
    print("=" * 78)

    def line(label, key, pct=True):
        m, s = summary.loc[key, "mean"], summary.loc[key, "std"]
        if pct:
            print(f"  {label:34s} {m * 100:7.2f}% +/- {s * 100:.2f}%")
        else:
            print(f"  {label:34s} {m:7.4f}  +/- {s:.4f}")

    line("XGBoost 5-class accuracy", "accuracy")
    line("XGBoost macro-F1", "macro_f1", pct=False)
    line("XGBoost macro-precision", "macro_precision", pct=False)
    line("XGBoost macro-recall", "macro_recall", pct=False)
    print()
    line("XGBoost FNR (attack vs normal)", "xgb_binary_fnr")
    line("XGBoost FPR (attack vs normal)", "xgb_binary_fpr")
    line("Fixed 11% FNR (attack vs normal)", "fixed11_binary_fnr")
    line("Fixed 11% FPR (attack vs normal)", "fixed11_binary_fpr")
    print("\n  Per-class F1:")
    for i, n in enumerate(CLASS_NAMES):
        line(f"    {n}", f"f1_class_{i}", pct=False)

    print("\n  Leakage audit (best single-feature one-vs-rest AUC, last seed):")
    for cls, a in audits[-1].items():
        print(f"    class {cls} ({CLASS_NAMES[cls]:22s}) AUC {a['best_auc']:.4f}  via {a['feature']}")
    print(f"    max across classes, all seeds: {df['max_single_feature_auc'].max():.4f}")

    with open(metrics_dir / "multiattack_leakage_audit.json", "w") as f:
        json.dump(
            {
                "nuisance_ranges": {k: list(v) for k, v in NUISANCE_RANGES.items()},
                "per_seed_max_single_feature_auc": df["max_single_feature_auc"].tolist(),
                "last_seed_audit": {str(k): v for k, v in audits[-1].items()},
            },
            f,
            indent=2,
        )

    print(f"\n  Total elapsed: {time.time() - t0:.0f}s")
    print("  Wrote multiattack_multiseed_per_seed.csv, _summary.csv, _leakage_audit.json")
    print("=" * 78)


if __name__ == "__main__":
    main()
