"""FastAPI backend for the QKD Security Operations Center dashboard.

Serves the static frontend in `frontend/` and exposes, as JSON:
  * the calibrated BB84 simulator (7 classes, config/physics.yaml) with the trained
    XGBoost model's per-block predictions,
  * SHAP attribution and per-detector threshold verdicts for any block,
  * benchmark, channel-generalisation, MDI-QKD and dataset/leakage results.

Every response that carries results includes a `provenance` block: the source file(s),
the seed(s) and whether the data is simulated or real. Nothing here invents numbers;
missing result files are reported as missing.

Run:  python run_dashboard.py       (or: uvicorn backend.server:app --reload)
"""

from __future__ import annotations

import copy
import json
import os
import sys
import uuid
import secrets
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import load_physics  # noqa: E402
from src.features.time_series import (  # noqa: E402
    FEATURE_CONTEXT_63,
    FEATURE_GROUPS_63,
    FEATURE_NAMES_63,
    CausalMultiAttackFeatureExtractor,
)
from src.simulator.calibrated import CLASS_KEYS, CLASS_NAMES, CalibratedBB84Simulator  # noqa: E402
from backend import hosted_sessions

FRONTEND = ROOT_DIR / "frontend"
METRICS = ROOT_DIR / "results" / "metrics"
PROCESSED = ROOT_DIR / "data" / "processed"
TRIALS = PROCESSED / "trial_sessions"
CKPT = PROCESSED / "checkpoints" / "xgb_multiattack.joblib"
PHYS = load_physics()["calibrated_bb84"]

SIM_LABEL = "Simulated data"
REAL_LABEL = "Real data (benign, Zenodo 21792844)"
SEEDS_MULTI = "seeds 42-46"

CLASS_SHORT = ["Normal", "Intercept-Resend", "PNS", "Blinding", "Time-Shift", "Trojan Horse", "Beam Splitting"]
CLASS_FULL = [
    "Normal channel (no eavesdropper)",
    "Fractional intercept-resend",
    "Photon number splitting (PNS)",
    "Detector blinding (fake states)",
    "Time-shift (detector efficiency mismatch)",
    "Trojan horse (simplified)",
    "Beam splitting",
]
CLASS_COLORS = ["#0d9488", "#dc2626", "#0284c7", "#7c3aed", "#d97706", "#be185d", "#4d7c0f"]

CLASS_PHYSICS = [
    {"what": "No eavesdropper. Errors come from optical misalignment (1-2%), detector dark counts "
             "and their afterpulses; loss comes from fibre (0.2 dB/km) and connectors.",
     "watch": "QBER sits at its baseline. The baseline still drifts, and about a third of sessions "
              "contain a benign loss step, so 'something changed' is not an attack by itself.",
     "hard": "Benign QBER rises with distance as dark counts take over, which is exactly where weak "
             "attacks hide."},
    {"what": "Eve intercepts a fraction of pulses, measures in a random basis and resends what she saw, "
             "tuning her resend intensity so Bob's click rate does not change.",
     "watch": "QBER rises by about a quarter of the intercepted fraction. This is the one attack that "
              "QBER statistics catch reliably.",
     "hard": "At a 5% interception rate the added error (~1.2 points) is comparable to benign drift."},
    {"what": "Eve counts photons without disturbing them, keeps one photon from every multi-photon pulse "
             "and forwards the rest losslessly, tuning forwarding so Bob's click rate matches the honest one.",
     "watch": "Nothing in single-intensity telemetry: no QBER rise and a matched click rate. Only decoy "
              "states (comparing yields at several intensities) expose it.",
     "hard": "Without decoy states this attack is undetectable by construction; low recall here is the "
             "physically correct result."},
    {"what": "Bright light keeps Bob's detectors in linear mode; Eve's trigger pulses make them click only "
             "when Bob's basis equals hers, reproducing her measured bit.",
     "watch": "QBER falls below the optical floor and the dark-count floor disappears.",
     "hard": "A threshold on QBER can never fire, because the attack makes the channel look cleaner. "
             "See the MDI-QKD tab for the structural fix."},
    {"what": "Eve shifts pulse arrival times so one detector of each pair is less efficient, learning "
             "which bit is more likely without measuring the photon.",
     "watch": "A drop in click rate on the shifted pulses; QBER barely changes.",
     "hard": "Benign loss steps produce the same click-rate drop."},
    {"what": "Eve injects light into Alice's modulator and reads its back-reflection. In this simplified "
             "model Bob only sees a tiny leak of Eve's light as extra noise.",
     "watch": "Essentially nothing at Bob. The real signature is at Alice's watchdog photodiode, which is "
              "not part of Bob's telemetry.",
     "hard": "Invisible to QBER-based monitoring by design."},
    {"what": "Eve taps a fraction of every pulse and replaces the fibre with a lower-loss link so Bob's "
             "received intensity is unchanged.",
     "watch": "Nothing, while the tap can be compensated.",
     "hard": "Undetectable from Bob's statistics; privacy amplification, not monitoring, defends against it."},
]

SCENARIOS = [
    {"id": "baseline", "cls": 0, "channel": "fiber", "onset": 76, "blocks": 150,
     "title": "Clean baseline", "tag": "No attacker",
     "headline": "Establish what normal looks like before judging anything else.",
     "look": "QBER at its drawn baseline with drift; Q_Z and Q_X track together."},
    {"id": "stealth", "cls": 1, "channel": "fiber", "onset": 76, "blocks": 150,
     "title": "Sub-threshold intercept-resend", "tag": "Real leakage, no 11% alarm",
     "headline": "QBER rises at onset but stays under the 11% line.",
     "look": "The classical threshold never fires; the temporal model usually catches the step."},
    {"id": "pns", "cls": 2, "channel": "fiber", "onset": 76, "blocks": 150,
     "title": "Gain-matched PNS", "tag": "Invisible without decoys",
     "headline": "Neither QBER nor click rate moves.",
     "look": "Expect the model to miss this; that is the honest result without decoy states."},
    {"id": "blinding", "cls": 3, "channel": "fiber", "onset": 76, "blocks": 150,
     "title": "Detector blinding", "tag": "The alarm is inverted",
     "headline": "QBER goes DOWN during the attack.",
     "look": "QBER drops below the misalignment floor; no threshold can catch a cleaner-looking channel."},
    {"id": "timeshift", "cls": 4, "channel": "fiber", "onset": 76, "blocks": 150,
     "title": "Time-shift", "tag": "Click rate drops",
     "headline": "A loss-like signature that benign loss steps can mimic.",
     "look": "Watch the click-rate trace rather than QBER."},
    {"id": "trojan", "cls": 5, "channel": "fiber", "onset": 76, "blocks": 150,
     "title": "Trojan horse (simplified)", "tag": "Seen at Alice, not Bob",
     "headline": "Bob's telemetry stays clean.",
     "look": "Only a tiny noise rise reaches Bob; a watchdog at Alice is the real defence."},
    {"id": "beamsplit", "cls": 6, "channel": "fiber", "onset": 76, "blocks": 150,
     "title": "Beam splitting", "tag": "Undetectable when compensated",
     "headline": "Eve taps light and hides the loss.",
     "look": "Nothing to see; privacy amplification handles this one."},
    {"id": "freespace", "cls": 1, "channel": "free_space", "onset": 76, "blocks": 150,
     "title": "Turbulent free-space link", "tag": "Where false positives come from",
     "headline": "Log-normal fading moves the click rate block to block.",
     "look": "The baseline itself wanders; this is where a fibre-trained model's false-alarm rate climbs."},
]

app = FastAPI(title="QKD Security Operations Center", version="3.0")

def _load_bb84_model():
    hosted = ROOT_DIR / "hosted_models"
    if os.getenv("VERCEL") == "1" and (hosted / "manifest.json").exists():
        from backend.hosted_models import validate_bundle
        manifest = validate_bundle(hosted)
        filename = manifest.get("bb84_checkpoint")
        if filename in manifest["checksums"]:
            return joblib.load(hosted / filename)
    if CKPT.exists():
        return joblib.load(CKPT)
    return None


_model = _load_bb84_model()
_extractor = CausalMultiAttackFeatureExtractor(window_size=50, ema_span=10)
_sessions: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
_blind_truth = OrderedDict()
_blind_lock = threading.Lock()
_MAX_SESSIONS = 32


def _remember(df: pd.DataFrame) -> str:
    sid = uuid.uuid4().hex
    if hosted_sessions.enabled():
        # Explanations need the computed feature row, not the simulated truth.
        hosted_sessions.save("bb84-features", sid, df.to_json(orient="split", double_precision=15))
        return sid
    _sessions[sid] = df
    while len(_sessions) > _MAX_SESSIONS:
        _sessions.popitem(last=False)
    return sid


def _csv(name: str) -> Optional[List[Dict[str, Any]]]:
    p = METRICS / name
    if not p.exists():
        return None
    return pd.read_csv(p).replace({np.nan: None}).to_dict(orient="records")


def _json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _prov(files, seeds: str, label: str = SIM_LABEL) -> Dict[str, Any]:
    files = [files] if isinstance(files, str) else list(files)
    return {"label": label, "files": files, "seeds": seeds,
            "missing": [f for f in files if not (ROOT_DIR / f).exists()]}


MODEL_HELP = (
    "The XGBoost checkpoint data/processed/checkpoints/xgb_multiattack.joblib is "
    "missing. It is produced by the pipeline, not stored in git (*.joblib is "
    "gitignored). Generate it with:  python scripts/06_generate_multiattack_dataset.py  "
    "then  python scripts/07_train_multiattack_classifiers.py"
)


def _package(df: pd.DataFrame, cls: int, p_eve: float, onset: int, extra: Optional[Dict] = None) -> Dict[str, Any]:
    """Turn a feature frame plus model predictions into the wire format."""
    if _model is None and "pred_class" not in df.columns:
        raise HTTPException(503, MODEL_HELP)
    n_cls = len(CLASS_NAMES)
    if _model is not None and "pred_class" not in df.columns:
        if _model.n_features_in_ != len(FEATURE_NAMES_63):
            raise HTTPException(503,'Checkpoint does not match the deployed 63-feature schema')
        probs = _model.predict_proba(df[FEATURE_NAMES_63].values)
        df = df.copy()
        df["pred_class"] = probs.argmax(axis=1)
        for k in range(probs.shape[1]):
            df[f"prob_class_{k}"] = probs[:, k]
    n_prob = sum(1 for c in df.columns if c.startswith("prob_class_"))

    post = df[df["block_index"] >= onset]
    alarms = post.loc[post["pred_class"] != 0, "block_index"]
    delta_t = int(alarms.min() - onset) if len(alarms) else None
    correct = post.loc[(post['pred_class']==cls) & (post['pred_class']!=0),'block_index']
    post_acc = float((post["pred_class"] == cls).mean()) if len(post) else None
    fp_rate = float((df["pred_class"] != 0).mean()) if cls == 0 else None
    has_ci = "qber_ci_low" in df.columns

    sid = _remember(df)
    out = {
        "session_id": sid,
        "class_id": cls,
        "class_name": CLASS_SHORT[cls],
        "class_full": CLASS_FULL[cls],
        "p_eve": round(float(p_eve), 4),
        "onset": onset,
        "blocks": df["block_index"].astype(int).tolist(),
        "qber": df["qber_mean"].round(6).tolist(),
        "qber_block": (df["qber_block"].round(6).tolist() if "qber_block" in df else None),
        "ci_low": (df["qber_ci_low"].round(6).tolist() if has_ci else None),
        "ci_high": (df["qber_ci_high"].round(6).tolist() if has_ci else None),
        "sifted_bits": (df["sifted_bits_block"].astype(int).tolist() if "sifted_bits_block" in df else None),
        "qber_z": df["qber_rectilinear"].round(6).tolist(),
        "qber_x": df["qber_diagonal"].round(6).tolist(),
        "efficiency": df["transmission_efficiency"].round(8).tolist(),
        "asymmetry": df["basis_asymmetry"].round(6).tolist(),
        "pred": df["pred_class"].astype(int).tolist(),
        "probs": [[round(float(df[f"prob_class_{k}"].iloc[i]), 5) for k in range(n_prob)]
                  for i in range(len(df))],
        "summary": {
            "mean_qber_post": float(post["qber_mean"].mean()),
            "final_efficiency": float(df["transmission_efficiency"].iloc[-1]),
            "final_asymmetry": float(df["basis_asymmetry"].iloc[-1]),
            "delta_t": delta_t,
            "correct_class_latency": int(correct.min()-onset) if len(correct) and cls else None,
            "latency_definition": "delta_t: any non-normal prediction; correct_class_latency: correct attack classification; misses null",
            "first_alarm": int(alarms.min()) if len(alarms) else None,
            "threshold_fires": bool((post["qber_mean"] > 0.11).any()),
            "post_accuracy": post_acc,
            "false_positive_rate": fp_rate,
        },
        "provenance": {"label": SIM_LABEL, "engine": "calibrated (config/physics.yaml)",
                       "model": "data/processed/checkpoints/xgb_multiattack.joblib"},
    }
    if extra:
        out["provenance"].update(extra)
    return out


def _simulate(cls: int, channel: str, blocks: int, onset: int, seed: int, physics: Optional[Dict] = None,
              distance_km: Optional[float] = None, intensity: Optional[float] = None,
              detector: Optional[str] = None):
    phys = physics or PHYS
    if detector and detector != "auto":
        phys = copy.deepcopy(phys)
        phys["detector"]["types"] = [detector]
    sim = CalibratedBB84Simulator(phys, random_state=seed)
    rng = np.random.default_rng(seed)
    sp = sim.draw_session_physics(rng, CLASS_KEYS[cls], channel, blocks=blocks)
    if distance_km is not None and channel == "fiber":
        # Keep the session's drawn connector/optics loss; replace only the fibre part.
        # (Free-space loss is dominated by geometry, so distance is not overridden there.)
        alpha = phys["channel"]["fiber"]["attenuation_db_per_km"]
        sp.loss_db = alpha * float(distance_km) + (sp.loss_db - alpha * sp.distance_km)
        sp.distance_km = float(distance_km)
    if intensity is not None and cls != 0:
        sp.intensity = float(intensity)
    seq = sim.simulate_session(1, cls, channel_type=channel, blocks_per_session=blocks,
                               attack_start_block=onset, session_seed=seed, physics=sp)
    df = _extractor.extract_from_session(seq)
    if df.empty:
        raise HTTPException(400, "session too short for the W=50 window")
    return df, sp


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.get("/api/meta")
def meta():
    ms = _csv("multiattack_multiseed_summary.csv")
    headline = {}
    if ms:
        lookup = {r[list(r.keys())[0]]: r for r in ms}
        for k in ("accuracy", "macro_f1", "xgb_binary_fnr", "xgb_binary_fpr", "fixed11_binary_fnr",
                  "max_single_feature_auc"):
            if k in lookup:
                headline[k] = {c: lookup[k].get(c) for c in ("mean", "std", "ci95_low", "ci95_high", "n_seeds")}
    att = PHYS["attacks"]
    return {
        "class_names": CLASS_SHORT,
        "class_full": CLASS_FULL,
        "class_colors": CLASS_COLORS,
        "class_physics": CLASS_PHYSICS,
        "scenarios": SCENARIOS,
        "feature_groups": {k: v for k, v in FEATURE_GROUPS_63.items()},
        "controls": {
            "fiber_distance_km": PHYS["channel"]["fiber"]["distance_km"],
            "heldout_distance_km": PHYS["heldout"]["distance_km"],
            "detector_types": PHYS["detector"]["types"],
            "attack_intensity": {CLASS_KEYS.index(a): att[a]["intensity"] for a in att},
            "attack_heldout_intensity": {CLASS_KEYS.index(a): att[a]["heldout_intensity"] for a in att},
            "abort_qber": PHYS["asymptotic_abort_qber"],
        },
        "headline": headline,
        "headline_provenance": _prov("results/metrics/multiattack_multiseed_summary.csv", SEEDS_MULTI),
        "model_loaded": _model is not None,
        "n_classes": len(CLASS_NAMES),
    }


@app.get("/api/trial/{cls}")
def trial(cls: int):
    if not 0 <= cls < len(CLASS_NAMES):
        raise HTTPException(400, f"class must be 0-{len(CLASS_NAMES) - 1}")
    p = TRIALS / f"trial_class{cls}.csv"
    if not p.exists():
        raise HTTPException(404, "trial data missing - run scripts/11_mdi_protocol_comparison.py")
    df = pd.read_csv(p)
    if _model is not None and int(getattr(_model, "n_classes_", len(CLASS_NAMES))) != \
            sum(1 for c in df.columns if c.startswith("prob_class_")):
        df = df.drop(columns=[c for c in df.columns if c.startswith("prob_class_")] + ["pred_class"], errors="ignore")
    return _package(df, cls, float(df["session_p_eve"].iloc[0]), int(df["attack_start_block"].iloc[0]),
                    {"file": f"data/processed/trial_sessions/trial_class{cls}.csv"})


class SimReq(BaseModel):
    cls: int = Field(default=0, ge=0, le=6)
    channel: str = "fiber"
    blocks: int = Field(default=150, ge=100, le=250)
    onset: int = Field(default=76, ge=55, le=200)
    seed: int = Field(default=42, ge=0, le=10**9)


class BlindSimReq(BaseModel):
    channel: str = "fiber"
    blocks: int = Field(default=150, ge=100, le=250)
    seed: int = Field(default=42, ge=0, le=10**9)


class CustomSimReq(BaseModel):
    protocol: str = "bb84"                   # "bb84" or "decoy_bb84"
    channel: str = "fiber"
    distance_km: float = Field(default=30.0, ge=0.5, le=150.0)
    cls: int = Field(default=1, ge=0, le=6)
    intensity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    detector: str = "auto"                   # "auto", "ingaas_spad", "snspd"
    blocks: int = Field(default=150, ge=100, le=250)
    onset: int = Field(default=76, ge=55, le=200)
    seed: int = Field(default=42, ge=0, le=10**9)


def _check_channel(ch):
    if ch not in ("fiber", "free_space"):
        raise HTTPException(400, "channel must be fiber or free_space")


@app.post("/api/simulate/blind")
def simulate_blind(req: BlindSimReq):
    _check_channel(req.channel)
    seed = secrets.randbelow(10**9)
    rng = np.random.default_rng(seed)
    n = len(CLASS_NAMES)
    probs = np.array([0.25] + [0.75 / (n - 1)] * (n - 1))
    cls = int(rng.choice(range(n), p=probs))
    onset = int(rng.integers(55, min(req.blocks - 20, 95)))
    df, sp = _simulate(cls, req.channel, req.blocks, onset, seed)
    pkg = _package(df, cls, sp.intensity, onset)
    pkg["blind"] = True
    if hosted_sessions.enabled():
        hosted_sessions.save("bb84-reveal", pkg["session_id"], json.dumps(pkg))
    else:
        with _blind_lock:
            _blind_truth[pkg['session_id']] = copy.deepcopy(pkg)
            while len(_blind_truth)>_MAX_SESSIONS:
                _blind_truth.popitem(last=False)
    for key in ('class_id','class_name','class_full','p_eve','onset'):
        pkg.pop(key,None)
    pkg['summary']={k:v for k,v in pkg['summary'].items() if k in ('final_efficiency','final_asymmetry')}
    pkg['provenance'].pop('seed',None)
    return pkg


@app.post('/api/simulate/reveal/{session_id}')
def reveal_blind(session_id: str):
    if hosted_sessions.enabled():
        payload = hosted_sessions.read("bb84-reveal", session_id)
        value = json.loads(payload) if payload is not None else None
    else:
        with _blind_lock:
            value=_blind_truth.get(session_id)
    if value is None:
        raise HTTPException(404,'Blind session expired; rerun simulation')
    return value


@app.post("/api/simulate")
def simulate(req: SimReq):
    _check_channel(req.channel)
    onset = min(req.onset, req.blocks - 20)
    df, sp = _simulate(req.cls, req.channel, req.blocks, onset, req.seed)
    return _package(df, req.cls, sp.intensity, onset, {"seed": req.seed})


@app.post("/api/simulate/custom")
def simulate_custom(req: CustomSimReq):
    _check_channel(req.channel)
    if req.protocol not in ("bb84", "decoy_bb84"):
        raise HTTPException(400, "protocol must be bb84 or decoy_bb84")
    if req.detector not in ("auto", "ingaas_spad", "snspd"):
        raise HTTPException(400, "detector must be auto, ingaas_spad or snspd")
    phys = copy.deepcopy(PHYS)
    phys["source"]["decoy"]["enabled"] = req.protocol == "decoy_bb84"
    onset = min(req.onset, req.blocks - 20)
    df, sp = _simulate(req.cls, req.channel, req.blocks, onset, req.seed, physics=phys,
                       distance_km=req.distance_km if req.channel == "fiber" else None,
                       intensity=req.intensity, detector=req.detector)
    lo, hi = PHYS["channel"]["fiber"]["distance_km"]
    pkg = _package(df, req.cls, sp.intensity, onset, {
        "seed": req.seed, "protocol": req.protocol,
        "note": ("Decoy intensities are simulated, but the deployed model uses only the 63 base features."
                 if req.protocol == "decoy_bb84" else None),
        "out_of_training_range": bool(req.channel == "fiber" and not lo <= req.distance_km <= hi),
    })
    pkg["physics"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in sp.summary().items()}
    pkg['capabilities']={'feature_schema':FEATURE_NAMES_63,'decoy_aware_classifier':False,
                         'simulated_decoy_states':req.protocol=='decoy_bb84'}
    return pkg


@app.get("/api/explain")
def explain(session_id: str, block: int):
    if hosted_sessions.enabled():
        from io import StringIO
        payload = hosted_sessions.read("bb84-features", session_id)
        df = pd.read_json(StringIO(payload), orient="split", precise_float=True) if payload is not None else None
    else:
        df = _sessions.get(session_id)
    if df is None:
        raise HTTPException(404, "session expired - re-run the scenario")
    if _model is None:
        raise HTTPException(503, MODEL_HELP)
    row = df[df["block_index"] == block]
    if row.empty:
        raise HTTPException(404, "block not in session")

    x = row[FEATURE_NAMES_63].values.reshape(1, -1)
    pred = int(row["pred_class"].iloc[0])
    sv = _tree_explainer().shap_values(x)
    vals = sv[pred][0] if isinstance(sv, list) else sv[0, :, pred]

    group_of = {f: g for g, feats in FEATURE_GROUPS_63.items() for f in feats}
    order = np.argsort(np.abs(vals))[::-1]
    features = [{"rank": r + 1, "feature": FEATURE_NAMES_63[i], "shap": float(vals[i]),
                 "value": float(x[0, i]), "group": group_of[FEATURE_NAMES_63[i]],
                 "context": FEATURE_CONTEXT_63[FEATURE_NAMES_63[i]]}
                for r, i in enumerate(order)]
    groups = {}
    for g, feats in FEATURE_GROUPS_63.items():
        idx = [FEATURE_NAMES_63.index(f) for f in feats]
        groups[g] = float(np.abs(vals[idx]).sum())
    tot = sum(groups.values()) or 1.0
    groups = {k: v / tot for k, v in groups.items()}
    n_prob = sum(1 for c in row.columns if c.startswith("prob_class_"))
    return {
        "block": block, "pred": pred, "pred_name": CLASS_SHORT[pred],
        "probs": [float(row[f"prob_class_{k}"].iloc[0]) for k in range(n_prob)],
        "top_features": features[:14], "features": features, "groups": groups,
        "thresholds": _threshold_checks(row.iloc[0]),
        "provenance": {"label": SIM_LABEL, "model": "data/processed/checkpoints/xgb_multiattack.joblib"},
    }


_explainer = None


def _tree_explainer():
    """Build the SHAP explainer once; parsing the XGBoost trees costs ~0.8 s per build."""
    global _explainer
    if _explainer is None:
        import shap
        _explainer = shap.TreeExplainer(_model)
    return _explainer


HOEFFDING_ALPHA = 0.01      # config.yaml models.hoeffding.alpha (target FPR <= 1%)


def _threshold_checks(r: pd.Series) -> List[Dict[str, Any]]:
    """Does this block's evidence cross each detector's alarm threshold?"""
    abort = float(PHYS["asymptotic_abort_qber"])
    q = float(r["qber_mean"])
    pulses = float(PHYS["source"]["pulses_per_block"])
    n_bits = max(float(r["key_generation_rate"]) * pulses * _extractor.window_size, 1.0)
    e0 = _extractor.base_qber
    hoeff = e0 + float(np.sqrt(np.log(1.0 / HOEFFDING_ALPHA) / (2.0 * n_bits)))
    p_attack = 1.0 - float(r["prob_class_0"])
    pred = int(r["pred_class"])

    def check(name, metric, value, threshold, note, exceeded=None, unit="pct"):
        hit = bool(value > threshold) if exceeded is None else bool(exceeded)
        return {"name": name, "metric": metric, "value": value, "threshold": threshold,
                "exceeded": hit, "margin": value - threshold, "unit": unit, "note": note}

    out = [
        check("Classical 11% abort", "Window QBER (W=50)", q, abort, "Shor-Preskill BB84 abort limit"),
        check("Z-basis 11% abort", "Rectilinear QBER", float(r["qber_rectilinear"]), abort, "Same limit, Z basis only"),
        check("X-basis 11% abort", "Diagonal QBER", float(r["qber_diagonal"]), abort, "Same limit, X basis only"),
        check(f"Hoeffding dynamic (α={HOEFFDING_ALPHA:g})", "Window QBER (W=50)", q, hoeff,
              f"Assumes {e0:.0%} baseline, {n_bits:,.0f} bits"),
    ]
    if "qber_ci_low" in r.index and pd.notna(r["qber_ci_low"]):
        out.append(check("Finite-size 11% (Wilson)", "Block QBER 95% lower bound", float(r["qber_ci_low"]), abort,
                         f"Block of {int(r.get('sifted_bits_block', 0)):,} sifted bits"))
    out.append(check("XGBoost temporal detector", "P(attack) = 1 − P(normal)", p_attack, 0.5,
                     "Alarm when argmax class ≠ Normal", exceeded=pred != 0, unit="prob"))
    return out


@app.get("/api/mdi")
def mdi():
    return {
        "attack": _csv("mdi_detector_attack_comparison.csv"),
        "key_rate": _csv("mdi_key_rate_vs_distance.csv"),
        "provenance": {
            "attack": _prov("results/metrics/mdi_detector_attack_comparison.csv", "none (analytic)",
                            "Analytic model (no sampling)"),
            "key_rate": _prov("results/metrics/mdi_key_rate_vs_distance.csv", "none (analytic)",
                              "Analytic model (asymptotic key rates)"),
        },
    }


@app.get("/api/benchmarks")
def benchmarks():
    files = {
        "overall": "multiattack_overall_benchmarks.csv",
        "binary": "multiattack_binary_benchmarks.csv",
        "per_class": "multiattack_per_class_f1.csv",
        "ablation": "multiattack_feature_ablation.csv",
        "multiseed": "multiattack_multiseed_per_seed.csv",
        "summary": "multiattack_multiseed_summary.csv",
        "comparison": "multiattack_model_comparison.csv",
        "per_class_multiseed": "multiattack_per_class_metrics_multiseed.csv",
        "confusion": "multiattack_confusion_matrix_multiseed.csv",
        "latency": "multiattack_latency_vs_intensity.csv",
        "reliability": "multiattack_reliability.csv",
        "heldout": "multiattack_heldout_generalization.csv",
        "per_class_multiseed_decoy": "multiattack_per_class_metrics_multiseed_decoy.csv",
        "comparison_decoy": "multiattack_model_comparison_decoy.csv",
        "trials": "trial_session_summary.csv",
    }
    out = {k: _csv(v) for k, v in files.items()}
    cm = METRICS / files["confusion"]
    if cm.exists():
        m = pd.read_csv(cm, index_col=0)
        out["confusion"] = {"labels": list(m.columns), "values": m.values.tolist()}
    out["provenance"] = {k: _prov(f"results/metrics/{v}",
                                  "seed 42" if k in ("overall", "binary", "per_class", "ablation") else SEEDS_MULTI)
                         for k, v in files.items()}
    return out


@app.get("/api/channel")
def channel():
    return {
        "calibrated": _csv("multiattack_channel_generalization.csv"),
        "legacy": _csv("cross_channel_generalization.csv"),
        "provenance": {
            "calibrated": _prov("results/metrics/multiattack_channel_generalization.csv", SEEDS_MULTI),
            "legacy": _prov("results/metrics/cross_channel_generalization.csv",
                            "seed 42 (legacy 3-class pipeline)"),
        },
    }


@app.get("/api/data")
def data():
    sidecars = {}
    for name in ("train_multiattack", "test_multiattack", "heldout_generalization"):
        m = _json(PROCESSED / f"{name}.meta.json")
        if m:
            m.pop("features", None)
            sidecars[name] = m
    sess = PROCESSED / "sessions_multiattack.csv"
    session_stats = None
    if sess.exists():
        s = pd.read_csv(sess)
        session_stats = {
            "by_split_class": s.groupby(["split", "class_id"]).size().unstack(fill_value=0).to_dict(orient="index"),
            "distance_km": {sp: np.histogram(g["distance_km"], bins=12, range=(0, 120))[0].tolist()
                            for sp, g in s.groupby("split")},
            "distance_bins": np.linspace(0, 120, 13).tolist(),
            "detector_types": s["detector_type"].value_counts().to_dict(),
        }
    return {
        "sidecars": sidecars,
        "sessions": session_stats,
        "dataset_leakage": _json(METRICS / "multiattack_dataset_leakage_audit.json"),
        "multiseed_leakage": _json(METRICS / "multiattack_leakage_audit.json"),
        "real_calibration": _csv("real_vs_sim_qber_calibration.csv"),
        "real_ood": _json(METRICS / "real_data_ood_summary.json"),
        "class_names": CLASS_SHORT,
        "provenance": {
            "sidecars": _prov(["data/processed/train_multiattack.meta.json", "data/processed/test_multiattack.meta.json",
                               "data/processed/heldout_generalization.meta.json"], "seed 42"),
            "dataset_leakage": _prov("results/metrics/multiattack_dataset_leakage_audit.json", "seed 42"),
            "multiseed_leakage": _prov("results/metrics/multiattack_leakage_audit.json", SEEDS_MULTI),
            "real": _prov(["results/metrics/real_vs_sim_qber_calibration.csv",
                           "results/metrics/real_data_ood_summary.json"], "simulated side: seed 7", REAL_LABEL),
        },
    }


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": _model is not None,
            "detail": None if _model is not None else MODEL_HELP}


# --------------------------------------------------------------------------
# Static frontend (mounted last so /api/* wins)
# --------------------------------------------------------------------------
class _RevalidatingStatic(StaticFiles):
    """Static files that browsers must revalidate (ETag) so UI updates show on reload."""

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


from backend.source_side import router as source_router
app.include_router(source_router)

if FRONTEND.exists():
    app.mount("/static", _RevalidatingStatic(directory=str(FRONTEND)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND / "index.html"), headers={"Cache-Control": "no-cache"})
else:
    @app.get("/")
    def missing():
        return JSONResponse({"error": "frontend/ not found"}, status_code=500)
