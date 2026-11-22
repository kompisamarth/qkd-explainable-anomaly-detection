"""Multi-seed evaluation of RF, XGBoost and SVM-RBF against threshold baselines.

For every seed a fresh dataset is generated (calibrated engine, session-level split,
plus a held-out generalisation set with unseen attack intensities and distances), the
three models are trained, and every metric is reported as mean, std and a 95% Student-t
confidence interval over seeds. FNR (missed attacks, attack-vs-normal) is the primary
security metric.

Outputs in results/metrics/ (all simulated data):
  multiattack_multiseed_per_seed.csv       one row per seed (XGBoost columns keep their
                                            original names; rf_/svm_ prefixes for the others)
  multiattack_multiseed_summary.csv        mean / std / ci95 per metric
  multiattack_model_comparison.csv         model x metric, mean / std / ci95 (incl. baselines)
  multiattack_per_class_metrics_multiseed.csv   precision / recall / F1 per class and model
  multiattack_confusion_matrix_multiseed.csv    XGBoost confusion matrix summed over seeds
  multiattack_reliability.csv              binary P(attack) reliability curve per model
  multiattack_latency_vs_intensity.csv     blocks to first correct alarm vs attack intensity
  multiattack_heldout_generalization.csv   same metrics on the held-out set
  multiattack_leakage_audit.json           per-class best single-feature AUC, every seed
and figures fig_model_comparison_multiseed.png, fig_reliability.png, fig_latency_vs_intensity.png.

    python scripts/10_multiseed_evaluation.py --seeds 42 43 44 45 46 --sessions-per-class 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.dataset import generate_sessions, leakage_audit, model_features, session_split
from src.evaluation.metrics import (binary_rates, detection_latency, expected_calibration_error,
                                    finite_size_threshold_alarm, mean_ci, reliability_curve)
from src.simulator.calibrated import CLASS_NAMES

METRICS = ROOT_DIR / "results" / "metrics"
FIGURES = ROOT_DIR / "results" / "figures"


def load_config() -> dict:
    """config.yaml merged with config/physics.yaml (see src/config.py)."""
    from src.config import load_config as _load
    return _load()


def build_models(cfg_ma, seed, n_classes):
    x = cfg_ma.get("xgboost", {})
    return {
        "xgb": xgb.XGBClassifier(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss",
                                 tree_method="hist", n_estimators=x.get("n_estimators", 250),
                                 max_depth=x.get("max_depth", 6), learning_rate=x.get("learning_rate", 0.08),
                                 subsample=x.get("subsample", 0.85), colsample_bytree=x.get("colsample_bytree", 0.8),
                                 random_state=seed, n_jobs=-1),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_leaf=5, n_jobs=-1,
                                     random_state=seed),
        # SVM-RBF: kernel cost grows with n^2, so it trains on a session-stratified subsample
        "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                                                   random_state=seed)),
    }


def subsample_by_session(df, max_rows, seed):
    if len(df) <= max_rows:
        return df
    sess = df["session_id"].unique()
    rng = np.random.default_rng(seed)
    keep = rng.choice(sess, size=max(2, int(len(sess) * max_rows / len(df))), replace=False)
    return df[df["session_id"].isin(set(keep))]


def score(y, prob, n_classes):
    pred = prob.argmax(axis=1)
    y_bin = y > 0
    p_att = 1.0 - prob[:, 0]
    rates = binary_rates(y_bin, pred > 0)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(n_classes), zero_division=0)
    present = [c for c in range(n_classes) if (y == c).any()]
    try:
        auroc_macro = roc_auc_score(y, prob[:, present] / prob[:, present].sum(1, keepdims=True),
                                    multi_class="ovr", labels=present)
    except ValueError:
        auroc_macro = np.nan
    out = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=range(n_classes), zero_division=0),
        "macro_precision": float(np.mean(p)), "macro_recall": float(np.mean(r)),
        "binary_fnr": rates["fnr"], "binary_fpr": rates["fpr"],
        "binary_auroc": roc_auc_score(y_bin, p_att) if 0 < y_bin.mean() < 1 else np.nan,
        "binary_auprc": average_precision_score(y_bin, p_att) if 0 < y_bin.mean() < 1 else np.nan,
        "macro_auroc_ovr": auroc_macro,
        "ece_top_label": expected_calibration_error(y, prob),
        "brier_binary": brier_score_loss(y_bin, p_att),
    }
    for c in range(n_classes):
        out[f"precision_class_{c}"], out[f"recall_class_{c}"], out[f"f1_class_{c}"] = p[c], r[c], f[c]
    return out, pred, p_att


def baseline_scores(df, abort):
    y_bin = df["block_class"].to_numpy(int) > 0
    out = {}
    for name, alarm in (("fixed11", df["qber_mean"].to_numpy(float) > abort),
                        ("fixed11_block", df["qber_block"].to_numpy(float) > abort),
                        ("finite11", finite_size_threshold_alarm(df, abort))):
        r = binary_rates(y_bin, alarm)
        out[f"{name}_binary_fnr"], out[f"{name}_binary_fpr"] = r["fnr"], r["fpr"]
    return out


def run_seed(seed, cfg, spc, held_spc, svm_rows, models_wanted):
    ma, phys = cfg["multiattack"], cfg["calibrated_bb84"]
    classes = list(ma.get("class_ids", range(len(CLASS_NAMES))))
    n_classes = len(CLASS_NAMES)
    common = dict(classes=classes, channel=ma.get("channel", "fiber"), blocks=int(ma["blocks_per_session"]),
                  onset=int(ma["attack_start_block"]), window=int(ma["window_size"]), ema=int(ma["ema_span"]),
                  physics=phys)
    t0 = time.time()
    pool, _ = generate_sessions(spc, seed, heldout=False, **common)
    train, test = session_split(pool, float(ma.get("test_split", 0.2)), seed)
    held, _ = generate_sessions(held_spc, seed + 1_000_003, heldout=True, first_session_id=10_000_000, **common)
    # Channel generalisation: fibre-trained models on free-space sessions (same classes)
    fso, _ = generate_sessions(held_spc, seed + 2_000_003, heldout=False, first_session_id=20_000_000,
                               **{**common, "channel": "free_space"})
    t_gen = time.time() - t0
    feats = model_features(train)
    abort = float(phys["asymptotic_abort_qber"])

    row = {"seed": seed, "train_rows": len(train), "test_rows": len(test), "heldout_rows": len(held),
           "train_sessions": train["session_id"].nunique()}
    held_row = {"seed": seed}
    chan_row = {"seed": seed}
    extras = {"per_class": [], "reliability": [], "latency": [], "cm": None}
    row.update(baseline_scores(test, abort))
    held_row.update(baseline_scores(held, abort))

    for name, model in build_models(ma, seed, n_classes).items():
        if name not in models_wanted:
            continue
        tr = subsample_by_session(train, svm_rows, seed) if name == "svm" else train
        t1 = time.time()
        model.fit(tr[feats].values, tr["block_class"].values)
        # predict_proba columns follow model.classes_; map them onto all class ids
        def proba(df):
            pr = np.zeros((len(df), n_classes))
            pr[:, model.classes_ if hasattr(model, "classes_") else range(n_classes)] = model.predict_proba(df[feats].values)
            return pr
        prob_te = proba(test)
        s, pred, p_att = score(test["block_class"].to_numpy(int), prob_te, n_classes)
        pre = "" if name == "xgb" else f"{name}_"
        for k, v in s.items():
            key = {"binary_fnr": "xgb_binary_fnr", "binary_fpr": "xgb_binary_fpr"}.get(k, k) if name == "xgb" else k
            row[pre + key] = v
        row[f"{name}_fit_seconds"] = time.time() - t1
        hs, _, _ = score(held["block_class"].to_numpy(int), proba(held), n_classes)
        for k, v in hs.items():
            held_row[f"{name}_{k}"] = v
        fs, _, _ = score(fso["block_class"].to_numpy(int), proba(fso), n_classes)
        for k in ("accuracy", "macro_f1", "binary_fnr", "binary_fpr", "binary_auroc"):
            chan_row[f"{name}_fiber_{k}"] = s[k]
            chan_row[f"{name}_free_space_{k}"] = fs[k]

        for c in range(n_classes):
            extras["per_class"].append({"seed": seed, "model": name, "class_id": c, "class_name": CLASS_NAMES[c],
                                        "precision": s[f"precision_class_{c}"], "recall": s[f"recall_class_{c}"],
                                        "f1": s[f"f1_class_{c}"]})
        rel = reliability_curve(test["block_class"].to_numpy(int) > 0, p_att)
        extras["reliability"].append(rel.assign(seed=seed, model=name))
        y = test["block_class"].to_numpy(int)
        lat = detection_latency(test, pred > 0, correct_class=(pred == y) | (y == 0))
        extras["latency"].append(lat.assign(seed=seed, model=name))
        if name == "xgb":
            extras["cm"] = confusion_matrix(y, pred, labels=range(n_classes))

    audit = leakage_audit(train, feats, n_classes, seed=seed)
    row["max_single_feature_auc"] = audit["max_auc"]
    row["leakage_flagged"] = audit["any_flagged"]
    print(f"  seed {seed}: XGB acc {row.get('accuracy', np.nan):.4f} macroF1 {row.get('macro_f1', np.nan):.4f} "
          f"FNR {row.get('xgb_binary_fnr', np.nan):.4f} | fixed-11% FNR {row['fixed11_binary_fnr']:.4f} "
          f"| gen {t_gen:.0f}s | max single-feature AUC {audit['max_auc']:.3f}", flush=True)
    for b, v in baseline_scores(fso, abort).items():
        chan_row[f"{b}_free_space"] = v
    extras["channel"] = chan_row
    return row, held_row, extras, audit


def summarise(df, skip=("seed",)):
    rows = {}
    for c in df.columns:
        if c in skip or not np.issubdtype(df[c].dtype, np.number):
            continue
        rows[c] = mean_ci(df[c].astype(float).values)
    return pd.DataFrame(rows).T[["mean", "std", "ci95_low", "ci95_high", "n_seeds"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--sessions-per-class", type=int, default=200)
    ap.add_argument("--heldout-sessions-per-class", type=int, default=50)
    ap.add_argument("--svm-max-rows", type=int, default=12000)
    ap.add_argument("--models", nargs="+", default=["xgb", "rf", "svm"])
    ap.add_argument("--decoy", action="store_true",
                    help="enable signal/decoy/vacuum intensities and the decoy_state features; outputs get a _decoy suffix")
    args = ap.parse_args()
    suffix = "_decoy" if args.decoy else ""
    if len(args.seeds) < 5:
        print("  NOTE: fewer than 5 seeds; confidence intervals will be wide.")

    cfg = load_config()
    if args.decoy:
        cfg["calibrated_bb84"]["source"]["decoy"]["enabled"] = True
    METRICS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"MULTI-SEED EVALUATION  seeds={args.seeds}  sessions/class={args.sessions_per_class}  models={args.models}")
    print("=" * 78, flush=True)

    rows, held_rows, per_class, rel, lat, audits, cm_sum, chan = [], [], [], [], [], {}, None, []
    for seed in args.seeds:
        r, h, ex, audit = run_seed(seed, cfg, args.sessions_per_class, args.heldout_sessions_per_class,
                                   args.svm_max_rows, args.models)
        rows.append(r); held_rows.append(h); per_class += ex["per_class"]
        rel += ex["reliability"]; lat += ex["latency"]; audits[str(seed)] = audit
        chan.append(ex["channel"])
        if ex["cm"] is not None:
            cm_sum = ex["cm"] if cm_sum is None else cm_sum + ex["cm"]

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(METRICS / f"multiattack_multiseed_per_seed{suffix}.csv", index=False)
    summary = summarise(per_seed)
    summary.to_csv(METRICS / f"multiattack_multiseed_summary{suffix}.csv")

    # Model comparison table (one row per model x metric)
    comp = []
    metric_map = {"accuracy": "accuracy", "macro_f1": "macro_f1", "binary_fnr": "binary_fnr",
                  "binary_fpr": "binary_fpr", "binary_auroc": "binary_auroc", "binary_auprc": "binary_auprc",
                  "macro_auroc_ovr": "macro_auroc_ovr", "ece_top_label": "ece_top_label",
                  "brier_binary": "brier_binary"}
    for m in args.models:
        for metric in metric_map:
            col = ({"binary_fnr": "xgb_binary_fnr", "binary_fpr": "xgb_binary_fpr"}.get(metric, metric)
                   if m == "xgb" else f"{m}_{metric}")
            if col in per_seed:
                comp.append({"model": m, "metric": metric, **mean_ci(per_seed[col].values)})
    for b, label in (("fixed11", "Fixed 11% (window mean, asymptotic)"),
                     ("fixed11_block", "Fixed 11% (single block, asymptotic)"),
                     ("finite11", "Finite-size 11% (Wilson lower bound)")):
        for metric in ("binary_fnr", "binary_fpr"):
            comp.append({"model": label, "metric": metric, **mean_ci(per_seed[f"{b}_{metric}"].values)})
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(METRICS / f"multiattack_model_comparison{suffix}.csv", index=False)

    pc = pd.DataFrame(per_class)
    pc_sum = pc.groupby(["model", "class_id", "class_name"]).agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std")).reset_index()
    pc_sum.to_csv(METRICS / f"multiattack_per_class_metrics_multiseed{suffix}.csv", index=False)
    if cm_sum is not None:
        pd.DataFrame(cm_sum, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            METRICS / f"multiattack_confusion_matrix_multiseed{suffix}.csv")

    rel_df = pd.concat(rel, ignore_index=True)
    rel_df.to_csv(METRICS / f"multiattack_reliability{suffix}.csv", index=False)

    lat_df = pd.concat(lat, ignore_index=True)
    att = lat_df[lat_df["class_id"] > 0].copy()
    att["intensity_bin"] = att.groupby("class_id")["intensity"].transform(
        lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"], duplicates="drop") if s.nunique() > 3 else "all")
    lat_sum = att.groupby(["model", "class_id", "intensity_bin"], observed=True).agg(
        intensity_mean=("intensity", "mean"), sessions=("session_id", "size"),
        detection_rate=("detected", "mean"), latency_mean=("latency_blocks", "mean"),
        latency_median=("latency_blocks", "median")).reset_index()
    lat_sum["class_name"] = lat_sum["class_id"].map(dict(enumerate(CLASS_NAMES)))
    far = lat_df.groupby("model")["pre_attack_false_alarm_rate"].agg(["mean", "std"]).reset_index()
    far.columns = ["model", "pre_attack_false_alarm_mean", "pre_attack_false_alarm_std"]
    lat_sum = lat_sum.merge(far, on="model")
    lat_sum.to_csv(METRICS / f"multiattack_latency_vs_intensity{suffix}.csv", index=False)

    held = pd.DataFrame(held_rows)
    held_sum = summarise(held)
    held_sum.to_csv(METRICS / f"multiattack_heldout_generalization{suffix}.csv")

    chan_df = pd.DataFrame(chan)
    chan_rows = []
    for m in args.models:
        for k in ("accuracy", "macro_f1", "binary_fnr", "binary_fpr", "binary_auroc"):
            for ch in ("fiber", "free_space"):
                col = f"{m}_{ch}_{k}"
                if col in chan_df:
                    chan_rows.append({"model": m, "channel": ch, "metric": k, **mean_ci(chan_df[col].values)})
    for b in ("fixed11", "fixed11_block", "finite11"):
        for k in ("binary_fnr", "binary_fpr"):
            chan_rows.append({"model": b, "channel": "free_space", "metric": k,
                              **mean_ci(chan_df[f"{b}_{k}_free_space"].values)})
    pd.DataFrame(chan_rows).to_csv(METRICS / f"multiattack_channel_generalization{suffix}.csv", index=False)

    with open(METRICS / f"multiattack_leakage_audit{suffix}.json", "w", encoding="utf-8") as f:
        json.dump({"alert_threshold_auc": 0.95, "computed_on": "train split of each seed",
                   "per_seed": audits}, f, indent=2)

    # ---- figures ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    for ax, metric, title in ((axes[0], "macro_f1", "Macro-F1 (7 classes)"), (axes[1], "binary_fnr", "Missed-attack rate (FNR)")):
        sub = comp_df[comp_df["metric"] == metric]
        ax.bar(sub["model"], sub["mean"], yerr=[sub["mean"] - sub["ci95_low"], sub["ci95_high"] - sub["mean"]],
               capsize=4, color="#0d9488")
        ax.set_title(title, fontsize=10); ax.tick_params(axis="x", rotation=25, labelsize=7); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Simulated data, {len(args.seeds)} seeds, 95% CI", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGURES / f"fig_model_comparison_multiseed{suffix}.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5), dpi=200)
    for m, g in rel_df.groupby("model"):
        gg = g.groupby("bin_low").agg(p=("mean_predicted", "mean"), o=("observed_frequency", "mean")).reset_index()
        ax.plot(gg["p"], gg["o"], "o-", ms=3, label=m)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Predicted P(attack)"); ax.set_ylabel("Observed attack frequency")
    ax.set_title("Reliability (simulated test, all seeds)", fontsize=10); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / f"fig_reliability{suffix}.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    sub = lat_sum[lat_sum["model"] == "xgb"]
    for c, g in sub.groupby("class_id"):
        ax.plot(g["intensity_mean"], g["latency_mean"], "o-", label=CLASS_NAMES[c])
    ax.set_xscale("log"); ax.set_xlabel("Attack intensity (class-specific units, log)")
    ax.set_ylabel("Mean blocks to first correct alarm"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("Detection latency vs attack intensity (XGBoost, simulated)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / f"fig_latency_vs_intensity{suffix}.png"); plt.close(fig)

    print("\n" + comp_df.pivot(index="model", columns="metric", values="mean").round(4).to_string())
    print("\nWrote multi-seed metrics and figures.")


if __name__ == "__main__":
    main()
