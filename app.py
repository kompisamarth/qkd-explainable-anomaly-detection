"""Quantum Key Distribution (BB84) — Real-Time Security Operations Center (SOC) Dashboard.

Interactive Streamlit application implementing:
- Real-time block-by-block QBER streaming and dual-basis monitoring
- Live 5-class XGBoost threat classification HUD
- Detection latency clock and autonomous hardware key-abort action
- On-the-fly SHAP explainability waterfall inspector
- Academic benchmark visualizer (Confusion Matrix, ROC curves, and Feature Ablation)
"""

import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from src.simulator.bb84_engine import BB84Simulator
from src.simulator.eve_attacks import EveAttacker
from src.features.time_series import (
    CausalMultiAttackFeatureExtractor,
    FEATURE_NAMES_63,
    FEATURE_GROUPS_63,
)

# Page configuration
st.set_page_config(
    page_title="QKD Security Operations Center (SOC)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0px; }
    .sub-title { font-size: 1.05rem; color: #4B5563; margin-bottom: 20px; }
    .metric-card { background-color: #F3F4F6; border-radius: 8px; padding: 12px; border-left: 5px solid #3B82F6; }
    .alert-banner-danger { background-color: #FEE2E2; border: 2px solid #EF4444; border-radius: 8px; padding: 15px; color: #991B1B; font-weight: bold; }
    .alert-banner-safe { background-color: #ECFDF5; border: 2px solid #10B981; border-radius: 8px; padding: 15px; color: #065F46; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

CLASS_NAMES = [
    "Normal Channel (No Eve)",
    "Fractional Intercept-Resend (IR)",
    "Photon Number Splitting (PNS)",
    "Fake State (Detector Blinding)",
    "Time-Shift Attack",
]

CLASS_SHORT = ["Normal", "Fractional IR", "PNS Attack", "Fake State", "Time-Shift"]
CLASS_COLORS = ["#10B981", "#EF4444", "#3B82F6", "#8B5CF6", "#F59E0B"]


@st.cache_resource
def load_trained_model():
    ckpt_path = ROOT_DIR / "data" / "processed" / "checkpoints" / "xgb_multiattack.joblib"
    if not ckpt_path.exists():
        st.error(f"Model checkpoint not found at: {ckpt_path}")
        return None
    return joblib.load(ckpt_path)


def main():
    st.markdown('<div class="main-title">🛡️ QKD Network Security Operations Center (SOC)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Physics-Informed Temporal QBER Anomaly Detection & Multi-Attack Classification for BB84 QKD | <i>IIT Mandi & DRDO Reference Architecture</i></div>',
        unsafe_allow_html=True,
    )

    model = load_trained_model()
    if model is None:
        st.stop()

    # -------------------------------------------------------------
    # SIDEBAR: Simulation & Attack Configuration
    # -------------------------------------------------------------
    st.sidebar.header("⚙️ QKD Link & Adversary Controls")

    channel_type = st.sidebar.selectbox("Channel Medium", ["Fiber Optic Link (0.2 dB/km)", "Free-Space Atmospheric Link"])
    channel_str = "fiber" if "Fiber" in channel_type else "free_space"

    attack_choice = st.sidebar.selectbox(
        "Adversary Strategy",
        CLASS_NAMES,
        index=1,
    )
    attack_class_id = CLASS_NAMES.index(attack_choice)

    # Dynamic attack parameter controls
    st.sidebar.subheader("Threat Parameters")
    if attack_class_id == 0:
        st.sidebar.info("Normal channel: No active eavesdropping.")
        param_val = 0.0
    elif attack_class_id == 1:
        param_val = st.sidebar.slider("Interception Rate (r_e)", min_value=0.05, max_value=0.35, value=0.20, step=0.01)
    elif attack_class_id == 2:
        param_val = st.sidebar.slider("WCP Mean Photon Number (μ)", min_value=0.10, max_value=0.60, value=0.30, step=0.02)
    elif attack_class_id == 3:
        param_val = st.sidebar.slider("Blinding Trigger Efficiency", min_value=0.70, max_value=1.00, value=0.95, step=0.01)
    elif attack_class_id == 4:
        param_val = st.sidebar.slider("Optical Delay Probability (p_shift)", min_value=0.30, max_value=0.95, value=0.70, step=0.05)

    blocks_count = st.sidebar.slider("Session Length (Blocks)", min_value=80, max_value=180, value=120, step=10)
    onset_block = st.sidebar.slider("Attack Onset Block (T_attack)", min_value=30, max_value=blocks_count - 20, value=60, step=5)

    sim_speed = st.sidebar.slider("Stream Animation Delay (sec)", min_value=0.0, max_value=0.20, value=0.02, step=0.01)
    run_sim = st.sidebar.button("🚀 Start Live QKD Transmission", type="primary", use_container_width=True)

    # Tabs interface
    tab_live, tab_explain, tab_benchmarks, tab_arch = st.tabs([
        "📡 Live Stream Monitor & Threat HUD",
        "🔍 SHAP Explainability Inspector",
        "📊 Academic Benchmarks & Figures",
        "📚 Threat Physics & Feature Guide",
    ])

    # -------------------------------------------------------------
    # TAB 1: Live Monitor & Threat HUD
    # -------------------------------------------------------------
    with tab_live:
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        kpi_qber = col_kpi1.empty()
        kpi_rate = col_kpi2.empty()
        kpi_asymm = col_kpi3.empty()
        kpi_status = col_kpi4.empty()

        alert_box = st.empty()

        col_plot1, col_plot2 = st.columns(2)
        chart_qber = col_plot1.empty()
        chart_bases = col_plot2.empty()

        st.subheader("🤖 Real-Time Multi-Class XGBoost Threat HUD")
        hud_bars = st.empty()
        abort_banner = st.empty()

        if run_sim:
            # Simulate session
            sim = BB84Simulator(raw_photons_per_block=1024, random_state=int(time.time() * 1000) % 1000000)
            extractor = CausalMultiAttackFeatureExtractor(window_size=50)

            with st.spinner("Initializing optical channel and quantum state generator..."):
                seq = sim.simulate_session(
                    session_id=1,
                    target_attack_class=attack_class_id,
                    channel_type=channel_str,
                    blocks_per_session=blocks_count,
                    attack_start_block=onset_block,
                    attack_mode="multiattack",
                )
                df_features = extractor.extract_from_session(seq)

            st.session_state["session_data"] = df_features
            st.session_state["session_seq"] = seq

            # Stream blocks starting from window size W=50
            w = 50
            history_blocks = []
            history_qber = []
            history_qz = []
            history_qx = []
            first_alarm_block = None

            for i in range(len(df_features)):
                row = df_features.iloc[i]
                cur_block = int(row["block_index"])
                cur_q = float(row["qber_mean"])
                cur_qz = float(row["qber_rectilinear"])
                cur_qx = float(row["qber_diagonal"])
                cur_asymm = float(row["basis_asymmetry"])
                cur_eff = float(row["transmission_efficiency"])

                history_blocks.append(cur_block)
                history_qber.append(cur_q)
                history_qz.append(cur_qz)
                history_qx.append(cur_qx)

                # Model prediction on current window
                x_vec = row[FEATURE_NAMES_63].values.reshape(1, -1)
                probs = model.predict_proba(x_vec)[0]
                pred_cls = int(np.argmax(probs))
                conf = float(probs[pred_cls]) * 100.0

                # KPI cards update
                kpi_qber.metric("Sifted QBER", f"{cur_q * 100:.2f}%", delta=f"{(cur_q - 0.03)*100:+.2f}% vs nominal")
                kpi_rate.metric("Transmission Efficiency", f"{cur_eff * 100:.1f}%")
                kpi_asymm.metric("Basis Asymmetry (|Qz - Qx|)", f"{cur_asymm * 100:.2f}%")

                if pred_cls == 0:
                    kpi_status.markdown('<div class="metric-card" style="border-left-color: #10B981;"><b>CHANNEL STATUS</b><br><span style="color:#10B981;">SECURE TRANSMISSION</span></div>', unsafe_allow_html=True)
                else:
                    kpi_status.markdown(f'<div class="metric-card" style="border-left-color: #EF4444;"><b>CHANNEL STATUS</b><br><span style="color:#EF4444;">ATTACK: {CLASS_SHORT[pred_cls].upper()}</span></div>', unsafe_allow_html=True)

                # Charts update
                # Plot 1: QBER with 11% threshold line
                fig1, ax1 = plt.subplots(figsize=(6, 3.2), dpi=100)
                ax1.plot(history_blocks, [q * 100 for q in history_qber], color="#1E3A8A", lw=2, label="Measured QBER (%)")
                ax1.axhline(11.0, color="#EF4444", ls="--", lw=1.8, label="11% Abort Threshold")
                ax1.axvline(onset_block, color="orange", ls=":", lw=1.5, label=f"Onset (Block {onset_block})")
                ax1.set_xlim(w, blocks_count)
                ax1.set_ylim(0, 16)
                ax1.set_xlabel("Transmission Block Index", fontsize=8)
                ax1.set_ylabel("QBER (%)", fontsize=8)
                ax1.set_title("Temporal QBER Stream vs. Static Threshold", fontsize=9, fontweight="bold")
                ax1.grid(True, alpha=0.3, ls="--")
                ax1.legend(loc="upper left", fontsize=7)
                fig1.tight_layout()
                chart_qber.pyplot(fig1)
                plt.close(fig1)

                # Plot 2: Dual Basis Qz vs Qx
                fig2, ax2 = plt.subplots(figsize=(6, 3.2), dpi=100)
                ax2.plot(history_blocks, [qz * 100 for qz in history_qz], color="#2563EB", lw=1.8, label="Q_Z (Rectilinear)")
                ax2.plot(history_blocks, [qx * 100 for qx in history_qx], color="#D97706", lw=1.8, label="Q_X (Diagonal)")
                ax2.axvline(onset_block, color="orange", ls=":", lw=1.5)
                ax2.set_xlim(w, blocks_count)
                ax2.set_ylim(0, 16)
                ax2.set_xlabel("Transmission Block Index", fontsize=8)
                ax2.set_ylabel("Basis QBER (%)", fontsize=8)
                ax2.set_title("Dual-Basis Error Tracking (Q_Z vs. Q_X)", fontsize=9, fontweight="bold")
                ax2.grid(True, alpha=0.3, ls="--")
                ax2.legend(loc="upper left", fontsize=7)
                fig2.tight_layout()
                chart_bases.pyplot(fig2)
                plt.close(fig2)

                # Threat HUD bars
                prob_df = pd.DataFrame({
                    "Threat Class": CLASS_SHORT,
                    "Probability (%)": [p * 100 for p in probs],
                })
                hud_bars.bar_chart(prob_df.set_index("Threat Class"))

                # Alert and Latency
                if pred_cls != 0:
                    if first_alarm_block is None:
                        first_alarm_block = cur_block
                    delta_t = max(0, first_alarm_block - onset_block)
                    alert_box.markdown(
                        f'<div class="alert-banner-danger">🚨 SECURITY COMPROMISED: <b>{CLASS_NAMES[pred_cls]}</b> flagged at Block {cur_block} | Latency ΔT = {delta_t} blocks (Confidence: {conf:.1f}%)</div>',
                        unsafe_allow_html=True,
                    )
                    abort_banner.error(f"🛑 AUTONOMOUS ACTION TRIGGERED: Sifted Key Distillation ABORTED at Block {cur_block} to prevent secret key leakage!")
                else:
                    alert_box.markdown(
                        '<div class="alert-banner-safe">✅ QKD CHANNEL SECURE: No eavesdropping signature detected. Sifted key privacy amplification active.</div>',
                        unsafe_allow_html=True,
                    )
                    abort_banner.success("🟢 SECURE STATUS: Hardware key generation pipeline active.")

                if sim_speed > 0:
                    time.sleep(sim_speed)

    # -------------------------------------------------------------
    # TAB 2: SHAP Explainability Inspector
    # -------------------------------------------------------------
    with tab_explain:
        st.subheader("🔍 Local SHAP Feature Attribution Inspector")
        st.write("Inspect exact physical feature contributions justifying the classifier's alert on any specific block.")

        if "session_data" in st.session_state:
            df_sess = st.session_state["session_data"]
            selected_block = st.slider("Select Block to Explain", min_value=int(df_sess["block_index"].min()), max_value=int(df_sess["block_index"].max()), value=int(df_sess["block_index"].max()))

            row_match = df_sess[df_sess["block_index"] == selected_block]
            if not row_match.empty:
                sample_x = row_match[FEATURE_NAMES_63].values.reshape(1, -1)
                sample_probs = model.predict_proba(sample_x)[0]
                sample_pred = int(np.argmax(sample_probs))

                st.info(f"Block **{selected_block}** | Predicted Class: **{CLASS_NAMES[sample_pred]}** (Confidence: {sample_probs[sample_pred]*100:.1f}%)")

                # Compute fast TreeExplainer on single instance
                import shap
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(sample_x)

                if isinstance(sv, list):
                    cls_sv = sv[sample_pred][0]
                else:
                    cls_sv = sv[0, :, sample_pred]

                shap_df = pd.DataFrame({
                    "feature": FEATURE_NAMES_63,
                    "shap_value": cls_sv,
                    "abs_shap": np.abs(cls_sv),
                }).sort_values("abs_shap", ascending=False).head(12)

                fig_shap, ax_shap = plt.subplots(figsize=(8, 4.5), dpi=150)
                colors = ["#EF4444" if val > 0 else "#3B82F6" for val in shap_df["shap_value"]]
                ax_shap.barh(shap_df["feature"][::-1], shap_df["shap_value"][::-1], color=colors[::-1], edgecolor="black", lw=0.5)
                ax_shap.axvline(0, color="black", lw=1)
                ax_shap.set_xlabel("SHAP Value (Impact on Attack Score)", fontweight="bold", fontsize=9)
                ax_shap.set_title(f"Top Features Driving Detection of {CLASS_SHORT[sample_pred]} at Block {selected_block}", fontweight="bold", fontsize=10)
                ax_shap.grid(True, alpha=0.3, ls="--")
                fig_shap.tight_layout()
                st.pyplot(fig_shap)
                plt.close(fig_shap)

                # Plain-English Physical Interpretation
                top_f = list(shap_df["feature"].head(3))
                st.markdown("#### 📖 Plain-Language Physical Attribution:")
                if sample_pred == 1:
                    st.write(f"- **Elevated Burst & Instability:** `{top_f[0]}` and `{top_f[1]}` indicated high temporal variance and localized step changes consistent with wave-function collapse from fractional intercept-resend.")
                elif sample_pred == 2:
                    st.write(f"- **Decoupled Channel Yield:** `{top_f[0]}` and `{top_f[1]}` showed anomalous key generation rate drop without corresponding QBER increase, matching multi-photon pulse splitting.")
                elif sample_pred == 3:
                    st.write(f"- **Classical Gating & Low Noise:** `{top_f[0]}` and `{top_f[1]}` exhibited abnormal error-gap uniformity and near-zero QBER consistent with bright-pulse detector blinding.")
                elif sample_pred == 4:
                    st.write(f"- **Basis Asymmetry:** `{top_f[0]}` and `{top_f[1]}` revealed sharp divergence between rectilinear and diagonal bases (|Q_Z - Q_X| >> 0), confirming time-shift efficiency mismatch.")
                else:
                    st.write(f"- **Nominal Channel Behavior:** All 63 features align with stationary Poissonian fiber noise.")
        else:
            st.info("Run a simulation in Tab 1 first to inspect SHAP attributions.")

    # -------------------------------------------------------------
    # TAB 3: Academic Benchmarks & Evaluation Viewer
    # -------------------------------------------------------------
    with tab_benchmarks:
        st.subheader("📊 Academic Benchmarks (IIT Mandi / IEEE Conference Replication)")

        col_b1, col_b2 = st.columns(2)
        cm_path = ROOT_DIR / "results" / "figures" / "fig_confusion_matrix_5class.png"
        if cm_path.exists():
            col_b1.image(str(cm_path), caption="Figure 4: 5x5 Normalized Confusion Matrix (XGBoost)", use_container_width=True)

        roc_path = ROOT_DIR / "results" / "figures" / "fig_roc_curves_5class.png"
        if roc_path.exists():
            col_b2.image(str(roc_path), caption="Figure 2: One-vs-Rest ROC Curves across all 5 classes", use_container_width=True)

        col_b3, col_b4 = st.columns(2)
        shap_path = ROOT_DIR / "results" / "figures" / "fig_shap_multiattack.png"
        if shap_path.exists():
            col_b3.image(str(shap_path), caption="Figure 5: Global Physics-Informed Feature Importance", use_container_width=True)

        abl_path = ROOT_DIR / "results" / "figures" / "fig_ablation_5class.png"
        if abl_path.exists():
            col_b4.image(str(abl_path), caption="Figure 6: Leave-One-Group-Out Feature Ablation Study", use_container_width=True)

        st.markdown("#### Benchmark Metric Tables")
        overall_csv = ROOT_DIR / "results" / "metrics" / "multiattack_overall_benchmarks.csv"
        if overall_csv.exists():
            st.write("**Table II: Overall Detection & False Negative Rate (FNR) vs. Conventional Baseline**")
            st.dataframe(pd.read_csv(overall_csv), use_container_width=True)

        per_class_csv = ROOT_DIR / "results" / "metrics" / "multiattack_per_class_f1.csv"
        if per_class_csv.exists():
            st.write("**Table III: Per-Class Precision, Recall, and F1-Scores**")
            st.dataframe(pd.read_csv(per_class_csv), use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: Threat Physics & Feature Guide
    # -------------------------------------------------------------
    with tab_arch:
        st.subheader("📚 Threat Taxonomy & 63-Feature Architecture")
        st.markdown(
            r"""
            ### The 5 Operational Regimes:
            1. **Class 0 — Normal Channel:** Standard fiber background noise ($e_0 \approx 3\%$), symmetric bases ($Q_Z \approx Q_X$).
            2. **Class 1 — Fractional Intercept-Resend (IR):** Intercepts $r_e \in [0.10, 0.30]$ pulses, inducing sub-threshold QBER ($5.5\% - 9.5\%$) with high temporal jump energy.
            3. **Class 2 — Photon Number Splitting (PNS):** Splitting multi-photon pulses ($n \ge 2$). **$\Delta\text{QBER} \approx 0$**, but decouples key generation rate and transmission efficiency.
            4. **Class 3 — Fake State Blinding:** Blinding SPADs into linear mode with classical bright triggers. Near-zero QBER ($\approx 0 - 1\%$), abnormal error-gap CV.
            5. **Class 4 — Time-Shift Attack:** Delaying pulses across detector efficiency mismatch. Induces severe basis asymmetry ($|Q_Z - Q_X| \gg 0$).

            ### The 5 Feature Categories (63 Features Total):
            - **Statistical (7):** Mean, variance, std, skewness, kurtosis, IQR, CV.
            - **Burst & Instability (10):** Spike density, burst duration, jump energy, drift slope, error-gap stats, burstiness index.
            - **Spectral & Temporal (12):** Autocorrelations (Lags 1-5), DFT band energies, spectral entropy, spectral centroid, Shannon entropy, EMA divergence.
            - **Basis-Dependent (12):** Rectilinear QBER, Diagonal QBER, basis asymmetry, burst energy per basis, variance ratio, cross-correlation.
            - **Channel Interaction & Multi-Scale (22):** Key generation rate, transmission efficiency, QBER-loss interaction, multi-scale rolling stats (W=10, 25, 50), loss anomaly index.
            """
        )


if __name__ == "__main__":
    main()
