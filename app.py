"""QKD Security Operations Center - interactive teaching and analysis dashboard.

Six tabs:
  1. Start Here      - the problem, in plain language, with the honest headline numbers
  2. Live Monitor    - scenario-driven BB84 streaming with the 5-class threat HUD
  3. Why? (SHAP)     - per-block feature attribution with physical reasoning
  4. MDI-QKD         - the protocol that structurally kills the detector attacks
  5. Benchmarks      - corrected results and the label-leakage story
  6. Threat Atlas    - attack taxonomy, feature guide, and honest limitations

Run with:  python -m streamlit run app.py
"""

import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from src.features.time_series import (
    FEATURE_GROUPS_63,
    FEATURE_NAMES_63,
    CausalMultiAttackFeatureExtractor,
)
from src.simulator.bb84_engine import NUISANCE_RANGES, BB84Simulator

st.set_page_config(
    page_title="QKD Security Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 2.1rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0; }
    .sub-title  { font-size: 1.0rem; color: #4B5563; margin-bottom: 18px; }
    /* Cards use a fixed light background, so every text colour inside them is
       pinned explicitly - otherwise they inherit Streamlit's dark-theme text
       colour and become invisible when the app runs in dark mode. */
    .card       { background:#F8FAFC; color:#111827; border-radius:10px;
                  padding:14px 16px; border-left:5px solid #3B82F6;
                  margin-bottom:10px; }
    .card b     { color:#111827; }
    .card-good  { border-left-color:#10B981; }
    .card-bad   { border-left-color:#EF4444; }
    .card-warn  { border-left-color:#F59E0B; }
    .big        { font-size:1.6rem; font-weight:700; color:#111827; }
    .muted      { color:#6B7280; font-size:0.86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

CLASS_NAMES = [
    "Normal Channel (no Eve)",
    "Fractional Intercept-Resend",
    "Photon Number Splitting (PNS)",
    "Fake State (detector blinding)",
    "Time-Shift Attack",
]
CLASS_SHORT = ["Normal", "Fractional IR", "PNS", "Fake State", "Time-Shift"]
CLASS_COLORS = ["#10B981", "#EF4444", "#3B82F6", "#8B5CF6", "#F59E0B"]

# What each class does physically, what to watch, and why it is hard.
CLASS_PHYSICS = {
    0: {
        "what": "No eavesdropper. Errors come from fiber depolarization, detector dark "
                "counts and residual polarization misalignment.",
        "watch": "QBER sits at its baseline and the two bases agree (Q_Z ~ Q_X).",
        "hard": "Nothing - but note the baseline itself varies from 0.5% to 6% between "
                "sessions, which is exactly what makes weak attacks hard to call.",
    },
    1: {
        "what": "Eve intercepts a fraction r_e of pulses, measures in a random basis and "
                "re-sends what she saw. Each interception has a 1/4 chance of flipping "
                "the sifted bit, so QBER rises by about r_e/4.",
        "watch": "QBER creeps up but stays under 11%. The giveaway is temporal: jump "
                "energy and variance rise at onset.",
        "hard": "At r_e = 0.05-0.10 the induced error is comparable to normal baseline "
                "drift, so a single block tells you nothing.",
    },
    2: {
        "what": "Eve splits multi-photon pulses, keeps one photon and forwards the rest "
                "untouched. She never measures the forwarded photon.",
        "watch": "QBER does NOT rise. What changes is the yield: transmission efficiency "
                "and key-generation rate drop out of proportion to distance.",
        "hard": "A QBER monitor is completely blind. You need the loss/error relationship, "
                "not the error alone.",
    },
    3: {
        "what": "Eve shines bright light to force Bob's detectors out of Geiger mode into "
                "classical linear mode, then injects faked states that make them click "
                "exactly when she wants.",
        "watch": "QBER goes near zero and error gaps become unnaturally regular.",
        "hard": "This is the worst case. It has been demonstrated experimentally to "
                "recover the ENTIRE key without raising QBER. Eve controls the monitor "
                "itself. See the MDI-QKD tab for the structural fix.",
    },
    4: {
        "what": "Eve shifts pulse arrival times so they land where Bob's two detectors "
                "have different efficiencies, biasing which basis gets recorded.",
        "watch": "Strong basis asymmetry: |Q_Z - Q_X| grows while the average QBER stays "
                "normal.",
        "hard": "The session-average QBER looks fine. You only see it by comparing the "
                "two bases against each other.",
    },
}

# Scenario presets: (class, channel, onset, blocks, headline, what to look for)
SCENARIOS = {
    "① Clean baseline - no attacker": dict(
        cls=0, channel="fiber", onset=76, blocks=150,
        headline="Establish what 'normal' looks like before judging anything else.",
        look="QBER flat at its (randomly drawn) baseline; Q_Z and Q_X track together; "
             "the HUD should stay on Normal.",
    ),
    "② Stealth attack under the 11% threshold": dict(
        cls=1, channel="fiber", onset=76, blocks=150,
        headline="The attack the whole project exists for: real leakage, no alarm.",
        look="QBER rises at onset but stays under the red 11% line, so the classical "
             "threshold never fires. The temporal model should still catch it.",
    ),
    "③ Silent attack with zero QBER rise (PNS)": dict(
        cls=2, channel="fiber", onset=76, blocks=150,
        headline="Proof that QBER alone is the wrong instrument.",
        look="The QBER trace barely moves. Watch Transmission Efficiency instead - the "
             "yield collapses while the error rate says everything is fine.",
    ),
    "④ Detector blinding - the key is simply stolen": dict(
        cls=3, channel="fiber", onset=76, blocks=150,
        headline="QBER goes DOWN during the attack. The alarm is inverted.",
        look="QBER drops toward zero because Eve's classical pulses are noise-free. "
             "No threshold on QBER can catch something that makes the channel look "
             "cleaner than normal.",
    ),
    "⑤ Time-shift - the two bases disagree": dict(
        cls=4, channel="fiber", onset=76, blocks=150,
        headline="Average QBER is normal; the asymmetry between bases is not.",
        look="Q_Z and Q_X separate visibly after onset while the mean sits in the "
             "normal band.",
    ),
    "⑥ Hard mode - attack on a drifting free-space link": dict(
        cls=1, channel="free_space", onset=76, blocks=150,
        headline="Atmospheric turbulence mimics an attack; this is where false "
                 "positives come from.",
        look="The baseline itself wanders. This is the regime where the fiber-trained "
             "model's false-positive rate climbs (README section 7.3).",
    ),
}

METRICS = ROOT_DIR / "results" / "metrics"
FIGURES = ROOT_DIR / "results" / "figures"
TRIALS = ROOT_DIR / "data" / "processed" / "trial_sessions"


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    p = ROOT_DIR / "data" / "processed" / "checkpoints" / "xgb_multiattack.joblib"
    return joblib.load(p) if p.exists() else None


@st.cache_data
def load_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_trial(cls: int):
    p = TRIALS / f"trial_class{cls}.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def simulate_session(cls, channel, blocks, onset, seed):
    """Live BB84 session on the SAME leakage-free distribution the model was trained on."""
    sim = BB84Simulator(raw_photons_per_block=1024, random_state=seed)
    seq = sim.simulate_session(
        session_id=1, target_attack_class=cls, channel_type=channel,
        blocks_per_session=blocks, attack_start_block=onset,
        attack_mode="multiattack", randomize_nuisance=True,
    )
    df = CausalMultiAttackFeatureExtractor(window_size=50, ema_span=10).extract_from_session(seq)
    return df, float(seq.p_eve)


def predict_frame(model, df):
    probs = model.predict_proba(df[FEATURE_NAMES_63].values)
    out = df.copy()
    out["pred_class"] = probs.argmax(axis=1)
    for k in range(5):
        out[f"prob_class_{k}"] = probs[:, k]
    return out


def card(title, value, sub="", kind=""):
    return (f'<div class="card {kind}"><div class="muted">{title}</div>'
            f'<div class="big">{value}</div><div class="muted">{sub}</div></div>')


# --------------------------------------------------------------------------
# TAB 1 - Start Here
# --------------------------------------------------------------------------
def tab_start():
    st.subheader("What problem is this solving?")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            """
BB84 decides whether someone is listening by watching a single number: the
**Quantum Bit Error Rate (QBER)**. If it crosses about **11%**, the link aborts.

That rule has a hole in it. An attacker who intercepts only *part* of the traffic
raises the QBER to roughly 5-9% - real leakage, no alarm. And several attacks are
worse than silent: **detector blinding actually lowers the QBER**, because the
attacker replaces noisy quantum detection with clean classical pulses.

This project replaces the single-number test with a **temporal** one: it watches
how the error rate *behaves over time* across 63 physics-derived features, and uses
XGBoost to name which of five regimes the link is in. Every alarm comes with a SHAP
explanation of which physical quantity triggered it.
            """
        )
        st.info(
            "**The honest headline.** The fixed 11% threshold misses **99.4%** of attack "
            "blocks. The temporal model misses **11.5%**. That is roughly a nine-fold "
            "reduction in false negatives - measured after removing a label-leakage bug "
            "that had previously inflated accuracy to 99.9%. See the Benchmarks tab."
        )
    with c2:
        st.markdown(card("Multi-class accuracy", "85.6% ± 1.4%",
                         "5 seeds, session-level split"), unsafe_allow_html=True)
        st.markdown(card("Missed attacks (FNR)", "11.5%",
                         "vs 99.4% for the 11% threshold", "card-good"),
                    unsafe_allow_html=True)
        st.markdown(card("Attacks no protocol stops", "4",
                         "source-side light injection", "card-warn"),
                    unsafe_allow_html=True)

    st.divider()
    st.subheader("Which protocol should actually be used?")
    st.markdown(
        """
The attacks sort by **where they live**, and that decides everything:

| Where the attack lives | Examples | Fixed by MDI-QKD? |
|---|---|---|
| **Detector side** | blinding / faked-state, efficiency mismatch, time-shift, backflash | **Yes - structurally** |
| **Channel** | intercept-resend, beam splitting, PNS | Detectable / decoy states |
| **Source side** | Trojan-horse, laser seeding, induced photorefraction, laser damage | **No - and nothing else does either** |

**MDI-QKD is the right backbone.** It makes the measurement device untrusted, so Eve
is *allowed* to own the detectors and still learns nothing. That erases the entire
detector-side group - including blinding, the one attack that defeats every
statistical monitor. The **MDI-QKD** tab simulates this and shows the crossover point.

**But no protocol closes the source side.** Every security proof - BB84, decoy, MDI,
even device-independent QKD - assumes no unwanted light leaves Alice's and Bob's labs.
Trojan-horse attacks break that assumption from *outside* the protocol. That gap is
where machine-learning monitoring earns its place, and it is the Phase 3 plan.
        """
    )
    with st.expander("How do I use this dashboard?"):
        st.markdown(
            """
1. **Live Monitor** - pick a scenario from the sidebar and press Run. Each scenario is
   chosen to expose a different blind spot of the classical threshold.
2. **Why? (SHAP)** - after a run, inspect any block and see exactly which physical
   feature drove the call.
3. **MDI-QKD** - see what happens when the attacker owns the measurement device, under
   BB84 versus MDI-QKD.
4. **Benchmarks** - the corrected numbers, and the story of the leakage bug.
5. **Threat Atlas** - the physics of each attack and the limitations of all of it.
            """
        )


# --------------------------------------------------------------------------
# TAB 2 - Live Monitor
# --------------------------------------------------------------------------
def tab_live(model, cfg):
    scen = SCENARIOS[cfg["scenario"]]
    st.subheader(cfg["scenario"])
    st.caption(scen["headline"])
    st.markdown(f'<div class="card card-warn"><b>What to look for:</b> {scen["look"]}</div>',
                unsafe_allow_html=True)

    if not cfg["run"]:
        st.info("Configure in the sidebar, then press **Run scenario**.")
        if "frame" not in st.session_state:
            return

    if cfg["run"]:
        if cfg["mode"] == "Replay trial data (instant)":
            raw = load_trial(cfg["cls"])
            if raw is None:
                st.error("No trial data found. Run `python scripts/11_mdi_protocol_comparison.py`.")
                return
            frame = raw if "pred_class" in raw.columns else predict_frame(model, raw)
            p_eve = float(frame["session_p_eve"].iloc[0])
            onset = int(frame["attack_start_block"].iloc[0])
        else:
            with st.spinner("Preparing quantum states, propagating, sifting..."):
                df, p_eve = simulate_session(cfg["cls"], cfg["channel"], cfg["blocks"],
                                             cfg["onset"], cfg["seed"])
                frame = predict_frame(model, df)
            onset = cfg["onset"]
        st.session_state.update(frame=frame, p_eve=p_eve, onset=onset, cls=cfg["cls"])

    frame = st.session_state["frame"]
    onset = st.session_state["onset"]
    p_eve = st.session_state["p_eve"]
    true_cls = st.session_state["cls"]

    # ---- KPI row ----
    post = frame[frame["block_index"] >= onset]
    alarms = post[post["pred_class"] != 0]["block_index"]
    delta_t = int(alarms.min() - onset) if len(alarms) else None
    final = frame.iloc[-1]

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(card("Mean QBER after onset", f"{post['qber_mean'].mean() * 100:.2f}%",
                     "11% is the classical abort line"), unsafe_allow_html=True)
    k2.markdown(card("Transmission efficiency", f"{final['transmission_efficiency'] * 100:.1f}%",
                     "the PNS giveaway"), unsafe_allow_html=True)
    k3.markdown(card("Basis asymmetry |Q_Z−Q_X|", f"{final['basis_asymmetry'] * 100:.2f}%",
                     "the time-shift giveaway"), unsafe_allow_html=True)
    if delta_t is None:
        k4.markdown(card("Detection latency", "no alarm", "model stayed on Normal",
                         "card-good" if true_cls == 0 else "card-bad"), unsafe_allow_html=True)
    else:
        k4.markdown(card("Detection latency ΔT", f"{delta_t} blocks",
                         f"first alarm at block {int(alarms.min())}", "card-bad"),
                    unsafe_allow_html=True)

    # ---- verdict ----
    thr_fires = bool((post["qber_mean"] > 0.11).any())
    verdict = st.columns(2)
    with verdict[0]:
        if thr_fires:
            st.error("**Classical 11% threshold: FIRED.** This attack was loud enough "
                     "for the textbook rule.")
        else:
            st.warning("**Classical 11% threshold: SILENT.** It never crossed 11%, so a "
                       "conventional link would have kept distilling the key.")
    with verdict[1]:
        if true_cls == 0:
            fp = float((frame["pred_class"] != 0).mean())
            (st.success if fp < 0.1 else st.error)(
                f"**Temporal model:** flagged {fp:.1%} of blocks on a benign link "
                f"(false positives).")
        elif delta_t is not None:
            acc = float((post["pred_class"] == true_cls).mean())
            st.success(f"**Temporal model: DETECTED** in {delta_t} blocks, and identified "
                       f"the correct attack type on {acc:.0%} of post-onset blocks.")
        else:
            st.error("**Temporal model: MISSED.** This one slipped through - see the "
                     "note below.")

    if true_cls != 0 and p_eve > 0 and p_eve < 0.12:
        st.caption(f"Note: this session drew a very weak attack strength (p_eve = {p_eve:.3f}). "
                   "Near the detection floor the model genuinely struggles, and that is "
                   "reported rather than hidden.")

    # ---- charts ----
    c1, c2 = st.columns(2)
    b = frame["block_index"].values

    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=110)
    ax.plot(b, frame["qber_mean"] * 100, color="#1E3A8A", lw=2, label="Window-mean QBER")
    ax.axhline(11, color="#EF4444", ls="--", lw=1.6, label="11% abort threshold")
    ax.axvline(onset, color="#F59E0B", ls=":", lw=1.8, label=f"Attack onset (block {onset})")
    if delta_t is not None:
        ax.axvline(int(alarms.min()), color="#10B981", ls="-.", lw=1.6, label="Model alarm")
    ax.set_xlabel("Block index", fontsize=8); ax.set_ylabel("QBER (%)", fontsize=8)
    ax.set_title("Temporal QBER vs the static threshold", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=7); fig.tight_layout()
    c1.pyplot(fig); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=110)
    ax.plot(b, frame["qber_rectilinear"] * 100, color="#2563EB", lw=1.7, label="Q_Z (rectilinear)")
    ax.plot(b, frame["qber_diagonal"] * 100, color="#D97706", lw=1.7, label="Q_X (diagonal)")
    ax.axvline(onset, color="#F59E0B", ls=":", lw=1.8)
    ax.set_xlabel("Block index", fontsize=8); ax.set_ylabel("Basis QBER (%)", fontsize=8)
    ax.set_title("Dual-basis tracking - separation means time-shift", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=7); fig.tight_layout()
    c2.pyplot(fig); plt.close(fig)

    c3, c4 = st.columns(2)
    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=110)
    ax.plot(b, frame["transmission_efficiency"] * 100, color="#8B5CF6", lw=1.8)
    ax.axvline(onset, color="#F59E0B", ls=":", lw=1.8)
    ax.set_xlabel("Block index", fontsize=8); ax.set_ylabel("Efficiency (%)", fontsize=8)
    ax.set_title("Transmission efficiency - the yield channel (PNS)", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.3, ls="--"); fig.tight_layout()
    c3.pyplot(fig); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=110)
    for k in range(5):
        ax.plot(b, frame[f"prob_class_{k}"], lw=1.5, color=CLASS_COLORS[k], label=CLASS_SHORT[k])
    ax.axvline(onset, color="#F59E0B", ls=":", lw=1.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Block index", fontsize=8); ax.set_ylabel("Class probability", fontsize=8)
    ax.set_title("Live classifier belief over time", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=7, ncol=3); fig.tight_layout()
    c4.pyplot(fig); plt.close(fig)

    with st.expander("Explain this scenario's physics"):
        p = CLASS_PHYSICS[true_cls]
        st.markdown(f"**What Eve is doing.** {p['what']}\n\n"
                    f"**What it looks like.** {p['watch']}\n\n"
                    f"**Why it is hard.** {p['hard']}")
        st.caption(f"This session drew attack strength p_eve = {p_eve:.3f}. Every session "
                   "also draws its own baseline QBER, channel loss, dark-count rate and "
                   "detector basis mismatch from overlapping ranges, so no class is "
                   "identifiable from one number.")


# --------------------------------------------------------------------------
# TAB 3 - SHAP
# --------------------------------------------------------------------------
def tab_shap(model):
    st.subheader("Why did the model say that?")
    st.caption("An autonomous abort decision has to be auditable. Pick any block and see "
               "which physical quantity drove the classification.")

    if "frame" not in st.session_state:
        st.info("Run a scenario in the **Live Monitor** tab first.")
        return

    frame = st.session_state["frame"]
    lo, hi = int(frame["block_index"].min()), int(frame["block_index"].max())
    blk = st.slider("Block to explain", lo, hi, min(hi, st.session_state["onset"] + 10))

    row = frame[frame["block_index"] == blk]
    if row.empty:
        return
    x = row[FEATURE_NAMES_63].values.reshape(1, -1)
    probs = np.array([float(row[f"prob_class_{k}"].iloc[0]) for k in range(5)])
    pred = int(probs.argmax())

    c1, c2 = st.columns([1, 2])
    c1.markdown(card("Model verdict", CLASS_SHORT[pred],
                     f"confidence {probs[pred] * 100:.1f}%",
                     "card-good" if pred == 0 else "card-bad"), unsafe_allow_html=True)
    c1.markdown(card("True class", CLASS_SHORT[st.session_state["cls"]],
                     "ground truth from the simulator"), unsafe_allow_html=True)

    with st.spinner("Computing SHAP attribution..."):
        import shap
        sv = shap.TreeExplainer(model).shap_values(x)
    cls_sv = sv[pred][0] if isinstance(sv, list) else sv[0, :, pred]

    sdf = pd.DataFrame({"feature": FEATURE_NAMES_63, "shap": cls_sv})
    sdf["abs"] = sdf["shap"].abs()
    top = sdf.sort_values("abs", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=130)
    cols = ["#EF4444" if v > 0 else "#3B82F6" for v in top["shap"]]
    ax.barh(top["feature"][::-1], top["shap"][::-1], color=cols[::-1], edgecolor="black", lw=0.4)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("SHAP value (push toward this class)", fontweight="bold", fontsize=9)
    ax.set_title(f"Block {blk}: what drove '{CLASS_SHORT[pred]}'", fontweight="bold", fontsize=10)
    ax.grid(alpha=0.3, ls="--"); fig.tight_layout()
    c2.pyplot(fig); plt.close(fig)

    # group-level attribution
    gsum = {g: float(sdf[sdf["feature"].isin(f)]["abs"].sum())
            for g, f in FEATURE_GROUPS_63.items()}
    total = sum(gsum.values()) or 1.0
    st.markdown("#### Which family of physics carried the decision?")
    gcols = st.columns(5)
    for (g, v), col in zip(sorted(gsum.items(), key=lambda kv: -kv[1]), gcols):
        col.metric(g.replace("_", " ").title(), f"{v / total * 100:.0f}%")

    st.markdown("#### In plain language")
    t3 = list(top["feature"].head(3))
    reasons = {
        0: f"`{t3[0]}` and `{t3[1]}` are sitting where an unattacked link sits: a stable "
           f"error rate, symmetric bases and a yield consistent with the channel loss.",
        1: f"`{t3[0]}` and `{t3[1]}` show the temporal signature of intercept-resend - "
           f"a step up in error level with raised variance and jump energy, while the "
           f"absolute QBER stays legal.",
        2: f"`{t3[0]}` and `{t3[1]}` point at the yield, not the error rate. The key rate "
           f"has fallen further than the loss budget allows, with no matching QBER rise - "
           f"the fingerprint of multi-photon splitting.",
        3: f"`{t3[0]}` and `{t3[1]}` show an unnaturally *clean*, regular channel. Blinded "
           f"detectors driven by classical pulses produce fewer and more evenly spaced "
           f"errors than real quantum detection ever does.",
        4: f"`{t3[0]}` and `{t3[1]}` capture disagreement between the two bases. The mean "
           f"error is unremarkable, but Z and X have diverged - detector efficiency "
           f"mismatch being exploited.",
    }
    st.write(reasons[pred])
    if pred != st.session_state["cls"]:
        st.warning(f"This block was **misclassified** (true: "
                   f"{CLASS_SHORT[st.session_state['cls']]}). Misses cluster right after "
                   "onset, before the W=50 window has filled with attacked blocks, and on "
                   "weak attack strengths.")


# --------------------------------------------------------------------------
# TAB 4 - MDI-QKD
# --------------------------------------------------------------------------
def tab_mdi():
    st.subheader("MDI-QKD: making the detectors untrusted")
    st.markdown(
        """
Detector blinding wins against BB84 for one structural reason: **Eve controls the
device that also computes the security check.** She fakes the key bits *and* the
monitor that is supposed to catch her.

Measurement-Device-Independent QKD (Lo, Curty & Qi, 2012) removes that. Alice and
Bob both become *transmitters*; an untrusted relay, Charlie, performs a Bell-state
measurement and publicly announces the result. Eve is **allowed** to be Charlie.
She still fails, because the security monitor - the X-basis phase error - is computed
from states Alice and Bob prepared themselves, in their own labs. She cannot forge it,
because she does not know what they sent.
        """
    )

    cmp_df = load_csv(METRICS / "mdi_detector_attack_comparison.csv")
    if cmp_df is None:
        st.warning("Run `python scripts/11_mdi_protocol_comparison.py` to generate this data.")
        return

    ctrl = st.slider("Fraction of the measurement device Eve controls", 0.0, 1.0, 1.0, 0.05)
    row = cmp_df.iloc[(cmp_df["p_control"] - ctrl).abs().argmin()]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### BB84 with blinded detectors")
        st.markdown(card("Key bits Eve learns", f"{row['bb84_eve_information'] * 100:.0f}%",
                         "she reads them directly", "card-bad"), unsafe_allow_html=True)
        st.markdown(card("Error rate you observe", f"{row['bb84_qber'] * 100:.2f}%",
                         "below 11% - no abort", "card-bad"), unsafe_allow_html=True)
        st.error("**No alarm.** Eve also controls the check basis, so every quantity you "
                 "could monitor is one she is forging.")
    with c2:
        st.markdown("#### MDI-QKD with Eve as Charlie")
        st.markdown(card("Z-basis error you observe", f"{row['mdi_qber'] * 100:.2f}%",
                         "she can keep this clean"), unsafe_allow_html=True)
        st.markdown(card("X-basis phase error", f"{row['mdi_check_error'] * 100:.1f}%",
                         "she cannot forge this",
                         "card-bad" if row["mdi_aborts"] else "card-good"),
                    unsafe_allow_html=True)
        if row["mdi_aborts"]:
            st.success("**Protocol aborts.** Measuring the pulses to learn Z bits destroys "
                       "their X-basis coherence. Alice and Bob detect it by comparing "
                       "Charlie's announcement against what they know they prepared.")
        else:
            st.info("Below the abort threshold - Eve is learning little at this control level.")

    first = cmp_df.loc[cmp_df["mdi_aborts"], "p_control"].min()
    st.markdown(f"**Crossover:** MDI-QKD trips its monitor once Eve controls just "
                f"**{first:.0%}** of the relay. BB84 never trips at any control fraction.")

    f9 = FIGURES / "fig9_detector_attack_comparison.png"
    if f9.exists():
        st.image(str(f9), use_container_width=True)

    st.divider()
    st.subheader("The price: rate, not distance")
    st.markdown(
        """
MDI-QKD's key rate scales with the **product** of the two arm transmittances, so it
falls as fast as BB84 with distance and does **not** beat the repeaterless (PLOB)
bound. That is TF-QKD's trick, not MDI's. **MDI-QKD buys security, not reach** - which
is the right trade when the threat you actually face is a blinded detector.
        """
    )
    kr = load_csv(METRICS / "mdi_key_rate_vs_distance.csv")
    if kr is not None:
        f8 = FIGURES / "fig8_mdi_vs_protocols.png"
        cc1, cc2 = st.columns([3, 2])
        if f8.exists():
            cc1.image(str(f8), use_container_width=True)
        show = kr[kr["distance_km"].isin([0, 50, 100, 200, 300, 400])].set_index("distance_km")
        cc2.dataframe(show.style.format("{:.3e}"), use_container_width=True)
        cc2.caption("Each 100 km costs MDI-QKD and BB84 a factor ~100, but SNS-TF-QKD "
                    "only ~10 (sqrt-eta scaling).")

    with st.expander("What MDI-QKD does NOT fix"):
        st.markdown(
            """
MDI-QKD closes the detector side completely. It does **nothing** for the source side:
Trojan-horse probing, laser seeding / injection locking, induced photorefraction in
lithium-niobate modulators, and laser-damage attacks that degrade an isolator to enable
the others.

Those attacks violate the **sealed-laboratory assumption** that *every* proof relies on,
device-independent QKD included. No protocol choice fixes them - only hardware
countermeasures plus runtime monitoring. That is precisely the Phase 3 proposal: keep
MDI-QKD as the backbone, and point the temporal ML at source-side telemetry
(watchdog back-reflection, optical isolation, laser SMSR/RIN, modulator bias drift),
because those attacks have slow, drifting, inherently temporal signatures.
            """
        )


# --------------------------------------------------------------------------
# TAB 5 - Benchmarks
# --------------------------------------------------------------------------
def tab_benchmarks():
    st.subheader("Corrected benchmarks")
    ms = load_csv(METRICS / "multiattack_multiseed_summary.csv")
    if ms is not None:
        ms = ms.set_index(ms.columns[0])

        def g(k, col="mean"):
            return float(ms.loc[k, col]) if k in ms.index else float("nan")

        c = st.columns(4)
        c[0].metric("Accuracy", f"{g('accuracy') * 100:.2f}%", f"± {g('accuracy','std')*100:.2f}%")
        c[1].metric("Macro-F1", f"{g('macro_f1'):.4f}", f"± {g('macro_f1','std'):.4f}")
        c[2].metric("Model FNR", f"{g('xgb_binary_fnr') * 100:.2f}%",
                    f"vs {g('fixed11_binary_fnr')*100:.1f}% for 11% rule", delta_color="inverse")
        c[3].metric("Max single-feature AUC", f"{g('max_single_feature_auc'):.3f}",
                    "leakage audit")

    st.error(
        "**What changed, and why you should trust these numbers more than the old ones.** "
        "The previously reported 99.93% accuracy was an artefact. Every class was simulated "
        "at a fixed channel transmittance of 1.0, so `transmission_efficiency` was *exactly "
        "1.000 with zero variance* for the Normal and Fractional-IR classes - a perfect "
        "giveaway. Each attack also had hard-coded strength constants. The model was partly "
        "memorising the simulator. Sessions now draw baseline QBER, channel loss, dark "
        "counts, detector basis mismatch and attack strength from **overlapping** ranges, "
        "so a benign link can look like a weak attack. Accuracy fell to 85.6%, and that is "
        "the number worth reporting."
    )

    t1, t2 = st.columns(2)
    ov = load_csv(METRICS / "multiattack_overall_benchmarks.csv")
    if ov is not None:
        t1.markdown("**Overall (single canonical run)**")
        t1.dataframe(ov, use_container_width=True)
    pc = load_csv(METRICS / "multiattack_per_class_f1.csv")
    if pc is not None:
        t2.markdown("**Per class**")
        t2.dataframe(pc, use_container_width=True)

    st.caption("Fake State is now the hardest class, which is physically right: blinding is "
               "designed to hold the error rate at baseline, and benign sessions are allowed "
               "to run as low as 0.5% QBER. Time-Shift stays easiest - its basis asymmetry is "
               "a genuine two-channel signature, not a simulator artefact.")

    st.divider()
    st.markdown("#### Which feature families actually matter?")
    ab = load_csv(METRICS / "multiattack_feature_ablation.csv")
    if ab is not None:
        fig, ax = plt.subplots(figsize=(8, 3.4), dpi=130)
        a = ab.sort_values("accuracy_drop_pct")
        ax.barh(a["display_name"], a["accuracy_drop_pct"], color="#2563EB",
                edgecolor="black", lw=0.5)
        ax.set_xlabel("Accuracy lost when this group is removed (%)", fontweight="bold", fontsize=9)
        ax.grid(alpha=0.3, ls="--", axis="x"); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.caption("With the leakage removed, **basis-dependent** features are the most "
                   "important group (-7.2%), then channel interaction (-5.4%). Before the "
                   "fix every group scored under 1%, because one leaked feature carried the "
                   "whole decision.")

    st.divider()
    cols = st.columns(2)
    for col, (f, cap) in zip(cols * 2, [
        ("fig_confusion_matrix_5class.png", "Confusion matrix (5 classes)"),
        ("fig_roc_curves_5class.png", "One-vs-rest ROC curves"),
        ("fig_shap_multiattack.png", "Global SHAP feature importance"),
        ("fig_ablation_5class.png", "Leave-one-group-out ablation"),
    ]):
        p = FIGURES / f
        if p.exists():
            col.image(str(p), caption=cap, use_container_width=True)

    with st.expander("A second, milder leak worth knowing about"):
        st.markdown(
            "`cv_multiattack_xgboost.json` reports ~90.6%, above the 86.3% held-out figure, "
            "because that cross-validation shuffles at **block** level and therefore splits "
            "overlapping windows from the same session across folds. The session-level "
            "held-out number is the one to quote."
        )


# --------------------------------------------------------------------------
# TAB 6 - Threat Atlas
# --------------------------------------------------------------------------
def tab_atlas():
    st.subheader("Threat atlas")
    st.markdown("**The five simulated regimes.** Select one to read its physics.")
    pick = st.radio("Class", CLASS_SHORT, horizontal=True, label_visibility="collapsed")
    i = CLASS_SHORT.index(pick)
    p = CLASS_PHYSICS[i]
    st.markdown(f'<div class="card"><b>What Eve does.</b> {p["what"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card card-warn"><b>Observable signature.</b> {p["watch"]}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="card card-bad"><b>Why it is hard.</b> {p["hard"]}</div>',
                unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Attacks this project does *not* yet simulate (Phase 3)")
    st.markdown(
        """
| Attack | Mechanism | Temporal signature to hunt |
|---|---|---|
| **Trojan-horse** | Eve probes Alice's transmitter and reads back-reflections | Watchdog photodiode power, phase-locked to the modulation frame |
| **Laser seeding** | Injected light seeds Alice's laser cavity | Step change in side-mode suppression ratio, RIN, wavelength |
| **Induced photorefraction** | Short-wavelength light builds space charge in the LiNbO₃ modulator | **Slow saturating drift** of modulator bias over minutes-hours |
| **Laser damage** | High-power laser permanently degrades an isolator | **Monotonic, irreversible** isolation loss - a precursor enabling the others |

These are the 2023-2026 frontier, and crucially they are **not covered by existing
security proofs**. They are also the natural target for temporal ML, because each has a
slow or step-like signature that no static threshold can see.
        """
    )

    st.divider()
    st.markdown("#### The 63 features")
    for g, feats in FEATURE_GROUPS_63.items():
        with st.expander(f"{g.replace('_', ' ').title()} ({len(feats)})"):
            st.write(", ".join(f"`{f}`" for f in feats))

    st.divider()
    st.markdown("#### Per-session randomization (what kills label leakage)")
    nz = pd.DataFrame([{"parameter": k, "min": v[0], "max": v[1]}
                       for k, v in NUISANCE_RANGES.items()])
    st.dataframe(nz, use_container_width=True, hide_index=True)
    st.caption("These ranges deliberately overlap across classes so a benign session can "
               "look like a weak attack.")

    st.divider()
    st.warning(
        """
**Limitations.** Everything here is simulated - there is no hardware and no measurement
from a physical QKD system. The fake-state, time-shift and PNS models reproduce the
qualitative signature of each attack rather than deriving it from a full device model.
MDI-QKD key rates are asymptotic (no finite-key correction), use the infinite-decoy
limit, and Y11/e11 are leading-order expressions. Free-space drift is a log-normal AR(1)
process on the QBER, not a turbulence model. Phase 3 is a specification, not a result.
        """
    )


# --------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-title">🛡️ QKD Security Operations Center</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Temporal-QBER anomaly detection for BB84, with '
                'explainability - and MDI-QKD as the structural answer to detector '
                'attacks</div>', unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("Model checkpoint missing. Run scripts 06 and 07 first.")
        st.stop()

    st.sidebar.header("Scenario")
    scenario = st.sidebar.selectbox("Choose what to demonstrate", list(SCENARIOS.keys()))
    s = SCENARIOS[scenario]
    mode = st.sidebar.radio("Data source",
                            ["Replay trial data (instant)", "Simulate live (slower)"])

    st.sidebar.markdown("---")
    if mode == "Simulate live (slower)":
        blocks = st.sidebar.slider("Session length (blocks)", 100, 200, s["blocks"], 10)
        onset = st.sidebar.slider("Attack onset block", 60, blocks - 20, s["onset"], 5)
        seed = st.sidebar.number_input("Random seed", 0, 10**6, 42, 1)
        channel = st.sidebar.selectbox(
            "Channel", ["fiber", "free_space"],
            index=0 if s["channel"] == "fiber" else 1)
    else:
        blocks, onset, seed, channel = s["blocks"], s["onset"], 42, s["channel"]
        st.sidebar.caption("Trial data is pre-generated from the same leakage-free "
                           "distribution the model was trained on. Switch to live "
                           "simulation to change parameters.")

    run = st.sidebar.button("▶ Run scenario", type="primary", use_container_width=True)
    st.sidebar.markdown("---")
    st.sidebar.caption("Accuracy 85.6% ± 1.4% (5 seeds). The 11% threshold misses 99.4% "
                       "of attack blocks; this model misses 11.5%.")

    cfg = dict(scenario=scenario, cls=s["cls"], channel=channel, blocks=blocks,
               onset=onset, seed=int(seed), mode=mode, run=run)

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🏠 Start Here", "📡 Live Monitor", "🔍 Why? (SHAP)",
        "🛰️ MDI-QKD", "📊 Benchmarks", "📚 Threat Atlas",
    ])
    with t1: tab_start()
    with t2: tab_live(model, cfg)
    with t3: tab_shap(model)
    with t4: tab_mdi()
    with t5: tab_benchmarks()
    with t6: tab_atlas()


if __name__ == "__main__":
    main()
