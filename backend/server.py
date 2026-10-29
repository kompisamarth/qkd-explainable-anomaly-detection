"""FastAPI backend for the QKD Security Operations Center dashboard.

Replaces the previous Streamlit app. Serves a static animated frontend from
`frontend/` and exposes the simulator, the trained 5-class model, SHAP
attribution and the MDI-QKD protocol comparison as JSON.

Run:  python run_dashboard.py       (or: uvicorn backend.server:app --reload)
"""

from __future__ import annotations

import sys
import uuid
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

from src.features.time_series import (  # noqa: E402
    FEATURE_GROUPS_63,
    FEATURE_NAMES_63,
    CausalMultiAttackFeatureExtractor,
)
from src.simulator.bb84_engine import NUISANCE_RANGES, BB84Simulator  # noqa: E402

FRONTEND = ROOT_DIR / "frontend"
METRICS = ROOT_DIR / "results" / "metrics"
TRIALS = ROOT_DIR / "data" / "processed" / "trial_sessions"
CKPT = ROOT_DIR / "data" / "processed" / "checkpoints" / "xgb_multiattack.joblib"

CLASS_NAMES = ["Normal Channel", "Fractional IR", "PNS Attack", "Fake State", "Time-Shift"]
CLASS_FULL = [
    "Normal Channel (no eavesdropper)",
    "Fractional Intercept-Resend",
    "Photon Number Splitting (PNS)",
    "Fake State (detector blinding)",
    "Time-Shift Attack",
]
CLASS_COLORS = ["#0d9488", "#dc2626", "#0284c7", "#7c3aed", "#d97706"]

CLASS_PHYSICS = [
    {
        "what": "No eavesdropper. Errors come from fiber depolarization, detector dark counts "
                "and residual polarization misalignment.",
        "watch": "QBER sits at its baseline and the two bases agree (Q_Z is close to Q_X).",
        "hard": "Nothing - but the baseline itself varies from 0.5% to 6% between sessions, "
                "which is exactly what makes weak attacks hard to call.",
    },
    {
        "what": "Eve intercepts a fraction of pulses, measures in a random basis and re-sends "
                "what she saw. Each interception has a 1/4 chance of flipping the sifted bit, "
                "so QBER rises by about r_e / 4.",
        "watch": "QBER creeps up but stays under 11%. The giveaway is temporal: jump energy "
                "and variance rise at onset.",
        "hard": "At r_e = 0.05-0.10 the induced error is comparable to normal baseline drift, "
                "so a single block tells you nothing.",
    },
    {
        "what": "Eve splits multi-photon pulses, keeps one photon and forwards the rest "
                "untouched. She never measures the forwarded photon.",
        "watch": "QBER does NOT rise. The yield changes: transmission efficiency and key rate "
                "drop out of proportion to distance.",
        "hard": "A QBER monitor is completely blind. You need the loss/error relationship, "
                "not the error alone.",
    },
    {
        "what": "Eve shines bright light to force Bob's detectors out of Geiger mode into "
                "classical linear mode, then injects faked states that make them click exactly "
                "when she wants.",
        "watch": "QBER goes near zero and error gaps become unnaturally regular.",
        "hard": "The worst case. Demonstrated experimentally to recover the ENTIRE key without "
                "raising QBER. Eve controls the monitor itself - see the MDI-QKD view.",
    },
    {
        "what": "Eve shifts pulse arrival times so they land where Bob's two detectors have "
                "different efficiencies, biasing which basis gets recorded.",
        "watch": "Strong basis asymmetry: |Q_Z - Q_X| grows while the average QBER stays normal.",
        "hard": "The session-average QBER looks fine. You only see it by comparing the two "
                "bases against each other.",
    },
]

SCENARIOS = [
    {"id": "baseline", "cls": 0, "channel": "fiber", "onset": 76, "blocks": 150,
     "title": "Clean baseline", "tag": "No attacker",
     "headline": "Establish what normal looks like before judging anything else.",
     "look": "QBER flat at its randomly drawn baseline; Q_Z and Q_X track together; the "
             "classifier should stay on Normal."},
    {"id": "stealth", "cls": 1, "channel": "fiber", "onset": 76, "blocks": 150,
     "title": "Sub-threshold stealth", "tag": "Real leakage, no alarm",
     "headline": "The attack this whole project exists for.",
     "look": "QBER rises at onset but stays under the 11% line, so the classical threshold "
             "never fires. The temporal model should still catch it."},
    {"id": "pns", "cls": 2, "channel": "fiber", "onset": 76, "blocks": 150,
     "title": "Zero-QBER splitting", "tag": "QBER is the wrong instrument",
     "headline": "Proof that watching the error rate alone cannot work.",
     "look": "The QBER trace barely moves. Watch transmission efficiency instead - the yield "
             "collapses while the error rate says everything is fine."},
    {"id": "blinding", "cls": 3, "channel": "fiber", "onset": 76, "blocks": 150,
     "title": "Detector blinding", "tag": "The alarm is inverted",
     "headline": "QBER goes DOWN during the attack.",
     "look": "QBER drops toward zero because Eve's classical pulses are noise-free. No "
             "threshold can catch something that makes the channel look cleaner than normal."},
    {"id": "timeshift", "cls": 4, "channel": "fiber", "onset": 76, "blocks": 150,
     "title": "Time-shift asymmetry", "tag": "The two bases disagree",
     "headline": "Average QBER is normal; the asymmetry between bases is not.",
     "look": "Q_Z and Q_X separate visibly after onset while the mean sits in the normal band."},
    {"id": "freespace", "cls": 1, "channel": "free_space", "onset": 76, "blocks": 150,
     "title": "Turbulent free-space link", "tag": "Where false positives come from",
     "headline": "Atmospheric turbulence mimics an attack.",
     "look": "The baseline itself wanders. This is the regime where a fiber-trained model's "
             "false-positive rate climbs."},
]

app = FastAPI(title="QKD Security Operations Center", version="2.0")

_model = joblib.load(CKPT) if CKPT.exists() else None
_extractor = CausalMultiAttackFeatureExtractor(window_size=50, ema_span=10)
_sessions: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
_MAX_SESSIONS = 32


def _remember(df: pd.DataFrame) -> str:
    sid = uuid.uuid4().hex[:12]
    _sessions[sid] = df
    while len(_sessions) > _MAX_SESSIONS:
        _sessions.popitem(last=False)
    return sid


def _csv(name: str) -> Optional[List[Dict[str, Any]]]:
    p = METRICS / name
    if not p.exists():
        return None
    return pd.read_csv(p).replace({np.nan: None}).to_dict(orient="records")


def _package(df: pd.DataFrame, cls: int, p_eve: float, onset: int) -> Dict[str, Any]:
    """Turn a feature frame plus model predictions into the wire format."""
    if _model is not None and "pred_class" not in df.columns:
        probs = _model.predict_proba(df[FEATURE_NAMES_63].values)
        df = df.copy()
        df["pred_class"] = probs.argmax(axis=1)
        for k in range(5):
            df[f"prob_class_{k}"] = probs[:, k]

    post = df[df["block_index"] >= onset]
    alarms = post.loc[post["pred_class"] != 0, "block_index"]
    delta_t = int(alarms.min() - onset) if len(alarms) else None
    post_acc = float((post["pred_class"] == cls).mean()) if len(post) else None
    fp_rate = float((df["pred_class"] != 0).mean()) if cls == 0 else None

    sid = _remember(df)
    return {
        "session_id": sid,
        "class_id": cls,
        "class_name": CLASS_NAMES[cls],
        "class_full": CLASS_FULL[cls],
        "p_eve": round(float(p_eve), 4),
        "onset": onset,
        "blocks": df["block_index"].astype(int).tolist(),
        "qber": df["qber_mean"].round(6).tolist(),
        "qber_z": df["qber_rectilinear"].round(6).tolist(),
        "qber_x": df["qber_diagonal"].round(6).tolist(),
        "efficiency": df["transmission_efficiency"].round(6).tolist(),
        "asymmetry": df["basis_asymmetry"].round(6).tolist(),
        "pred": df["pred_class"].astype(int).tolist(),
        "probs": [[round(float(df[f"prob_class_{k}"].iloc[i]), 5) for k in range(5)]
                  for i in range(len(df))],
        "summary": {
            "mean_qber_post": float(post["qber_mean"].mean()),
            "final_efficiency": float(df["transmission_efficiency"].iloc[-1]),
            "final_asymmetry": float(df["basis_asymmetry"].iloc[-1]),
            "delta_t": delta_t,
            "first_alarm": int(alarms.min()) if len(alarms) else None,
            "threshold_fires": bool((post["qber_mean"] > 0.11).any()),
            "post_accuracy": post_acc,
            "false_positive_rate": fp_rate,
        },
    }


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.get("/api/meta")
def meta():
    ms = _csv("multiattack_multiseed_summary.csv")
    headline = {}
    if ms:
        lookup = {r[list(r.keys())[0]]: r for r in ms}
        for k in ("accuracy", "macro_f1", "xgb_binary_fnr", "fixed11_binary_fnr",
                  "max_single_feature_auc"):
            if k in lookup:
                headline[k] = {"mean": lookup[k]["mean"], "std": lookup[k]["std"]}
    return {
        "class_names": CLASS_NAMES,
        "class_full": CLASS_FULL,
        "class_colors": CLASS_COLORS,
        "class_physics": CLASS_PHYSICS,
        "scenarios": SCENARIOS,
        "feature_groups": {k: v for k, v in FEATURE_GROUPS_63.items()},
        "nuisance_ranges": {k: list(v) for k, v in NUISANCE_RANGES.items()},
        "headline": headline,
        "model_loaded": _model is not None,
    }


@app.get("/api/trial/{cls}")
def trial(cls: int):
    if not 0 <= cls <= 4:
        raise HTTPException(400, "class must be 0-4")
    p = TRIALS / f"trial_class{cls}.csv"
    if not p.exists():
        raise HTTPException(404, "trial data missing - run scripts/11_mdi_protocol_comparison.py")
    df = pd.read_csv(p)
    return _package(df, cls, float(df["session_p_eve"].iloc[0]),
                    int(df["attack_start_block"].iloc[0]))


class SimReq(BaseModel):
    cls: int = Field(default=0, ge=0, le=4)
    channel: str = "fiber"
    blocks: int = Field(default=150, ge=100, le=250)
    onset: int = Field(default=76, ge=55, le=200)
    seed: int = Field(default=42, ge=0, le=10**9)


class BlindSimReq(BaseModel):
    channel: str = "fiber"
    blocks: int = Field(default=150, ge=100, le=250)
    seed: int = Field(default=42, ge=0, le=10**9)


@app.post("/api/simulate/blind")
def simulate_blind(req: BlindSimReq):
    import numpy as np
    if req.channel not in ("fiber", "free_space"):
        raise HTTPException(400, "channel must be fiber or free_space")
    rng = np.random.default_rng(req.seed + 7919)
    # Realistic prior: 25% Normal, 20% IR, 20% PNS, 20% Fake State, 15% Time-Shift
    cls = int(rng.choice([0, 1, 2, 3, 4], p=[0.25, 0.20, 0.20, 0.20, 0.15]))
    onset = int(rng.integers(55, min(req.blocks - 20, 95)))
    sim = BB84Simulator(raw_photons_per_block=1024, random_state=req.seed)
    seq = sim.simulate_session(
        session_id=1, target_attack_class=cls, channel_type=req.channel,
        blocks_per_session=req.blocks, attack_start_block=onset,
        attack_mode="multiattack", randomize_nuisance=True,
    )
    df = _extractor.extract_from_session(seq)
    if df.empty:
        raise HTTPException(400, "session too short for the W=50 window")
    pkg = _package(df, cls, float(seq.p_eve), onset)
    pkg["blind"] = True
    return pkg


@app.post("/api/simulate")
def simulate(req: SimReq):
    if req.channel not in ("fiber", "free_space"):
        raise HTTPException(400, "channel must be fiber or free_space")
    onset = min(req.onset, req.blocks - 20)
    sim = BB84Simulator(raw_photons_per_block=1024, random_state=req.seed)
    seq = sim.simulate_session(
        session_id=1, target_attack_class=req.cls, channel_type=req.channel,
        blocks_per_session=req.blocks, attack_start_block=onset,
        attack_mode="multiattack", randomize_nuisance=True,
    )
    df = _extractor.extract_from_session(seq)
    if df.empty:
        raise HTTPException(400, "session too short for the W=50 window")
    return _package(df, req.cls, float(seq.p_eve), onset)


@app.get("/api/explain")
def explain(session_id: str, block: int):
    df = _sessions.get(session_id)
    if df is None:
        raise HTTPException(404, "session expired - re-run the scenario")
    if _model is None:
        raise HTTPException(503, "model checkpoint not loaded")
    row = df[df["block_index"] == block]
    if row.empty:
        raise HTTPException(404, "block not in session")

    import shap
    x = row[FEATURE_NAMES_63].values.reshape(1, -1)
    pred = int(row["pred_class"].iloc[0])
    sv = shap.TreeExplainer(_model).shap_values(x)
    vals = sv[pred][0] if isinstance(sv, list) else sv[0, :, pred]

    order = np.argsort(np.abs(vals))[::-1][:14]
    top = [{"feature": FEATURE_NAMES_63[i], "shap": float(vals[i]),
            "value": float(x[0, i])} for i in order]

    groups = {}
    for g, feats in FEATURE_GROUPS_63.items():
        idx = [FEATURE_NAMES_63.index(f) for f in feats]
        groups[g] = float(np.abs(vals[idx]).sum())
    tot = sum(groups.values()) or 1.0
    groups = {k: v / tot for k, v in groups.items()}

    return {
        "block": block,
        "pred": pred,
        "pred_name": CLASS_NAMES[pred],
        "probs": [float(row[f"prob_class_{k}"].iloc[0]) for k in range(5)],
        "top_features": top,
        "groups": groups,
    }


@app.get("/api/mdi")
def mdi():
    return {
        "attack": _csv("mdi_detector_attack_comparison.csv"),
        "key_rate": _csv("mdi_key_rate_vs_distance.csv"),
    }


@app.get("/api/benchmarks")
def benchmarks():
    return {
        "overall": _csv("multiattack_overall_benchmarks.csv"),
        "per_class": _csv("multiattack_per_class_f1.csv"),
        "ablation": _csv("multiattack_feature_ablation.csv"),
        "multiseed": _csv("multiattack_multiseed_per_seed.csv"),
        "summary": _csv("multiattack_multiseed_summary.csv"),
        "trials": _csv("trial_session_summary.csv"),
    }


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": _model is not None}


# --------------------------------------------------------------------------
# Static frontend (mounted last so /api/* wins)
# --------------------------------------------------------------------------
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND / "index.html"))
else:
    @app.get("/")
    def missing():
        return JSONResponse({"error": "frontend/ not found"}, status_code=500)
