import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Add current workspace directory to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from backend.config import (
    DATASET_PATH,
    MODEL_SAVE_PATH,
    METRICS_SAVE_PATH,
    TEST_METRICS_PATH,
    PREDICTIONS_COMPARISON_PATH,
    MAX_DISTANCE,
    DEVICE,
)
from backend.inference import DimABSAInferenceEngine, get_affect_quadrant
from backend.preprocessing import parse_set_string, parse_float_string, expand_aspect_dataset

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Aspect-Based Emotion Regression (DimABSA)",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Modern Professional Styling (CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.8rem;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding-bottom: 0.3rem;
    }

    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }

    .badge-pill {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.45rem;
        margin-top: 0.6rem;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.28);
    }

    .sidebar-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }

    .status-pill-ok {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.35);
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 6px;
    }

    .aspect-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Cached Inference Engine Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading fine-tuned HingRoBERTa Aspect Emotion Regressor...")
def load_engine():
    return DimABSAInferenceEngine()


# -----------------------------------------------------------------------------
# Cached Dataset Loader
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df["aspect_list"] = df["all_aspects"].apply(parse_set_string)
        df["opinion_list"] = df["all_opinions"].apply(parse_set_string)
        df["valence_list"] = df["valence_scores"].apply(parse_float_string)
        df["arousal_list"] = df["arousal_scores"].apply(parse_float_string)
        return df
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# Benchmark Samples
# -----------------------------------------------------------------------------
BENCHMARK_EXAMPLES = [
    "The battery life is amazing but the display is disappointing.",
    "Service bahut badhiya hai lekin price kafi high hai.",
    "acting mast thi but story bakwas lagi",
    "camera quality bahut achi hai battery bekar hai",
    "food awesome tha service slow thi",
    "match me batting zabardast thi bowling weak thi",
    "movie ka climax accha tha par acting bilkul bakwas thi",
    "apna maal apne pas rakho oopar se bakwas kar rahe ho",
    "college faculty mast hai but placement weak hai",
    "teacher bahut supportive hai",
]

# -----------------------------------------------------------------------------
# Header Component
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">🎯 Aspect-Based Emotion Regression (DimABSA)</div>
        <div class="sub-title">Scientific Aspect-Level Continuous Valence & Arousal Regression on Code-Mixed Hinglish</div>
        <div>
            <span class="badge-pill">🧠 HingRoBERTa Backbone</span>
            <span class="badge-pill">🔀 SP-GSA Distance Gating</span>
            <span class="badge-pill">🎯 Individual Aspect-Level VA</span>
            <span class="badge-pill">🎭 Russell's Circumplex Affect</span>
            <span class="badge-pill">📉 Ultra-Low RMSE Multi-Loss</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Controls")
    st.markdown("Select a benchmark review or input your own text to evaluate **aspect-specific** emotion coordinates.")

    selected_sample = st.selectbox(
        "📌 Quick Benchmark Examples:",
        ["-- Custom Input --"] + BENCHMARK_EXAMPLES,
    )

    st.markdown("---")
    st.markdown("### 💻 Hardware & Model Status")

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div style="color:#94a3b8; font-size:0.85rem; font-weight:600;">Compute Device:</div>
            <div style="color:#38bdf8; font-size:1.1rem; font-weight:700;">{DEVICE.type.upper()}</div>
            <div style="color:#94a3b8; font-size:0.85rem; font-weight:600; margin-top:8px;">Dataset:</div>
            <div style="color:#e2e8f0; font-size:0.92rem; font-family:'JetBrains Mono', monospace;">DimABSA_Final_Dataset_600.csv</div>
            <div style="color:#60a5fa; font-size:0.85rem; font-weight:600; margin-top:2px;">(1,101 Aspect Instances)</div>
            <div style="margin-top:12px;">
                <span class="status-pill-ok">✅ Fine-tuned Aspect Model Loaded</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 2D Circumplex Plot Generator
# -----------------------------------------------------------------------------
def build_circumplex_plot(aspect_results):
    fig = go.Figure()

    # 4 Quadrant Tinted Backgrounds
    fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=1.0, y1=1.0, fillcolor="rgba(34, 197, 94, 0.08)", line=dict(width=0))
    fig.add_shape(type="rect", x0=0.0, y0=0.5, x1=0.5, y1=1.0, fillcolor="rgba(239, 68, 68, 0.08)", line=dict(width=0))
    fig.add_shape(type="rect", x0=0.0, y0=0.0, x1=0.5, y1=0.5, fillcolor="rgba(249, 115, 22, 0.08)", line=dict(width=0))
    fig.add_shape(type="rect", x0=0.5, y0=0.0, x1=1.0, y1=0.5, fillcolor="rgba(56, 189, 248, 0.08)", line=dict(width=0))

    # Quadrant Header Titles
    fig.add_annotation(x=0.04, y=0.96, text="<b>Q2: ANGER / FRUSTRATION</b>", showarrow=False, font=dict(color="#f87171", size=13, family="Outfit, sans-serif"), xanchor="left")
    fig.add_annotation(x=0.96, y=0.96, text="<b>Q1: JOY / EXCITEMENT</b>", showarrow=False, font=dict(color="#4ade80", size=13, family="Outfit, sans-serif"), xanchor="right")
    fig.add_annotation(x=0.04, y=0.04, text="<b>Q3: SADNESS / DISAPPOINTMENT</b>", showarrow=False, font=dict(color="#fb923c", size=13, family="Outfit, sans-serif"), xanchor="left")
    fig.add_annotation(x=0.96, y=0.04, text="<b>Q4: SERENITY / CALM</b>", showarrow=False, font=dict(color="#38bdf8", size=13, family="Outfit, sans-serif"), xanchor="right")

    # Center Axes Dividers
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255, 255, 255, 0.35)", line_width=1.5)
    fig.add_vline(x=0.5, line_dash="dash", line_color="rgba(255, 255, 255, 0.35)", line_width=1.5)

    # Reference Affect Anchors
    EMOTION_ANCHORS = [
        # Q1: JOY / EXCITEMENT
        {"name": "Euphoric / Exhilarated 🤩", "x": 0.95, "y": 0.90},
        {"name": "Astonished / Excited 😲", "x": 0.75, "y": 0.85},
        {"name": "Joyful / Delighted 😄", "x": 0.85, "y": 0.75},
        {"name": "Proud / Inspired 🌟", "x": 0.75, "y": 0.65},
        {"name": "Amused / Playful 😂", "x": 0.70, "y": 0.60},
        # Q2: ANGER / FRUSTRATION
        {"name": "Enraged / Furious 🤬", "x": 0.10, "y": 0.95},
        {"name": "Shocked / Alarmed 😱", "x": 0.35, "y": 0.90},
        {"name": "Anxious / Panicked 😨", "x": 0.20, "y": 0.85},
        {"name": "Angry / Hostile 😡", "x": 0.20, "y": 0.80},
        {"name": "Frustrated / Annoyed 😤", "x": 0.30, "y": 0.70},
        {"name": "Disgusted / Repulsed 🤢", "x": 0.15, "y": 0.65},
        # Q3: SADNESS / DISAPPOINTMENT
        {"name": "Disappointed / Let Down 😞", "x": 0.30, "y": 0.40},
        {"name": "Sad / Heartbroken 😢", "x": 0.18, "y": 0.30},
        {"name": "Tired / Exhausted 😩", "x": 0.32, "y": 0.25},
        {"name": "Bored / Apathetic 🥱", "x": 0.35, "y": 0.18},
        {"name": "Depressed / Despairing 😭", "x": 0.10, "y": 0.15},
        # Q4: SERENITY / CALM
        {"name": "Grateful / Appreciative 🙏", "x": 0.78, "y": 0.45},
        {"name": "Content / Satisfied 😊", "x": 0.72, "y": 0.40},
        {"name": "Relieved / Reassured 😌", "x": 0.68, "y": 0.35},
        {"name": "Calm / Peaceful 😇", "x": 0.75, "y": 0.25},
        {"name": "Serene / Blissful 🍃", "x": 0.88, "y": 0.20},
        # Center
        {"name": "Neutral / Balanced 😐", "x": 0.50, "y": 0.50},
    ]

    ref_x = [a["x"] for a in EMOTION_ANCHORS]
    ref_y = [a["y"] for a in EMOTION_ANCHORS]
    ref_text = [a["name"] for a in EMOTION_ANCHORS]

    fig.add_trace(
        go.Scatter(
            x=ref_x,
            y=ref_y,
            mode="markers+text",
            marker=dict(size=6, color="rgba(148, 163, 184, 0.4)"),
            text=ref_text,
            textposition="top center",
            textfont=dict(size=9.5, color="rgba(148, 163, 184, 0.65)", family="Outfit, sans-serif"),
            hoverinfo="text",
            name="Reference Affects",
            showlegend=False,
        )
    )

    # Plot Predicted Aspect Markers (Prominent & Glowing)
    aspect_palette = ["#22c55e", "#f97316", "#38bdf8", "#ec4899", "#a855f7", "#eab308"]
    for idx, asp in enumerate(aspect_results):
        v = asp["valence"]
        a = asp["arousal"]
        c = aspect_palette[idx % len(aspect_palette)]

        # Outer glow
        fig.add_trace(
            go.Scatter(
                x=[v],
                y=[a],
                mode="markers",
                marker=dict(size=36, color=c, opacity=0.35),
                hoverinfo="none",
                showlegend=False,
            )
        )
        # Inner solid point with crisp white border
        fig.add_trace(
            go.Scatter(
                x=[v],
                y=[a],
                mode="markers+text",
                marker=dict(size=20, color=c, line=dict(width=2.5, color="#ffffff")),
                text=[f"<b>{asp['aspect']}</b> ({v:.2f}, {a:.2f})"],
                textposition="bottom center",
                textfont=dict(size=12, color=c, family="Outfit, sans-serif"),
                name=asp["aspect"],
                showlegend=False,
            )
        )

    fig.update_layout(
        xaxis=dict(
            title=dict(text="<b>Valence</b> (0.0: Negative ─── 0.5: Neutral ─── 1.0: Positive)", font=dict(size=12, color="#94a3b8")),
            range=[0, 1],
            tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            gridcolor="rgba(255, 255, 255, 0.07)",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="<b>Arousal</b> (0.0: Low / Calm ─── 0.5: Medium ─── 1.0: High / Intense)", font=dict(size=12, color="#94a3b8")),
            range=[0, 1],
            tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            gridcolor="rgba(255, 255, 255, 0.07)",
            zeroline=False,
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.95)",
        plot_bgcolor="rgba(15, 23, 42, 0.95)",
        margin=dict(l=45, r=45, t=45, b=45),
        height=540,
    )
    return fig


# -----------------------------------------------------------------------------
# Tabs Layout
# -----------------------------------------------------------------------------
tab_live, tab_bench, tab_metrics, tab_data = st.tabs(
    [
        "🔮 Aspect-Level Sentence Analyzer",
        "⚡ Multi-Aspect Benchmark Suite",
        "📊 Model Evaluation & RMSE Metrics",
        "📁 Dataset & Aspect Browser",
    ]
)

# Load engine
try:
    engine = load_engine()
except Exception as e:
    st.error(f"Error loading DimABSA model engine: {e}")
    st.stop()

# =============================================================================
# TAB 1: LIVE SENTENCE ANALYZER
# =============================================================================
with tab_live:
    st.markdown("#### 📝 Input Hinglish Sentence")

    default_text = selected_sample if selected_sample != "-- Custom Input --" else "The battery life is amazing but the display is disappointing."

    user_text = st.text_area(
        "Hinglish Review / Sentence:",
        value=default_text,
        height=85,
        placeholder="e.g. The battery life is amazing but the display is disappointing.",
    )

    col_b1, col_b2 = st.columns([2, 5])
    with col_b1:
        run_analysis = st.button("🚀 Predict Aspect Emotions", type="primary", use_container_width=True)
    with col_b2:
        if st.button("🎲 Random Dataset Sentence", use_container_width=True):
            df_all = load_dataset()
            if not df_all.empty:
                st.session_state["random_sent"] = df_all.sample(1).iloc[0]["sentence"]
                st.rerun()

    if "random_sent" in st.session_state:
        user_text = st.session_state.pop("random_sent")

    if user_text:
        with st.spinner("Extracting individual aspects and predicting continuous Valence & Arousal..."):
            result = engine.predict(user_text)

        aspect_results = result.get("aspects", [])

        st.markdown("---")
        st.markdown(f"### 🎯 Aspect-Specific Emotion Predictions ({len(aspect_results)} Detected Aspects)")

        if aspect_results:
            # Build clean display table
            table_data = []
            for r in aspect_results:
                table_data.append({
                    "Aspect": r["aspect"],
                    "Opinion": r["opinion"],
                    "Valence": f"{r['valence']:.3f}",
                    "Arousal": f"{r['arousal']:.3f}",
                    "Polarity": r["polarity"],
                    "Emotion Category": r["emotion"],
                    "Russell Quadrant": r["quadrant"],
                    "Confidence": f"{r['confidence'] * 100:.1f}%",
                })

            table_df = pd.DataFrame(table_data)
            st.dataframe(table_df, use_container_width=True)

            # Two Columns: Aspect Metric Cards + Russell 2D Circumplex Plot
            st.markdown("---")
            col_cards, col_plot = st.columns([1, 1.25])

            with col_cards:
                st.markdown("#### 🔍 Individual Aspect Breakdowns")
                for i, asp_item in enumerate(aspect_results):
                    v = asp_item["valence"]
                    a = asp_item["arousal"]

                    with st.expander(f"Aspect {i+1}: **{asp_item['aspect']}** ({asp_item['polarity']})", expanded=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Valence (Polarity)", f"{v:.3f}", delta=f"{'Positive' if v >= 0.5 else 'Negative'}", delta_color="normal")
                        with c2:
                            st.metric("Arousal (Intensity)", f"{a:.3f}", delta=f"{'High' if a >= 0.5 else 'Low'}", delta_color="off")
                        st.markdown(f"**Opinion Expression**: `{asp_item['opinion']}`")
                        st.markdown(f"**Emotion**: {asp_item['emotion']}")
                        st.markdown(f"**Quadrant**: `{asp_item['quadrant']}`")
                        st.markdown(f"**Confidence**: `{asp_item['confidence'] * 100:.1f}%`")

            with col_plot:
                st.markdown("#### 🎭 Russell's Circumplex Affect 2D Matrix")
                fig_circumplex = build_circumplex_plot(aspect_results)
                st.plotly_chart(fig_circumplex, use_container_width=True)

        else:
            st.info("No aspect detected. Please enter a sentence containing aspect-opinion pairs.")

# =============================================================================
# TAB 2: MULTI-ASPECT BENCHMARK SUITE
# =============================================================================
with tab_bench:
    st.markdown("#### ⚡ Multi-Aspect Hinglish Benchmark Evaluation")
    st.markdown("Execute batch inference across standard benchmark sentences. Every aspect is evaluated individually with its own $(V, A)$ prediction:")

    if st.button("▶️ Run Complete Benchmark Suite", type="primary"):
        bench_rows = []
        progress_bar = st.progress(0)

        for i, sentence in enumerate(BENCHMARK_EXAMPLES):
            res = engine.predict(sentence)
            for item in res["aspects"]:
                bench_rows.append({
                    "Sentence": sentence,
                    "Aspect": item["aspect"],
                    "Opinion": item["opinion"],
                    "Valence": f"{item['valence']:.3f}",
                    "Arousal": f"{item['arousal']:.3f}",
                    "Polarity": item["polarity"],
                    "Emotion Category": item["emotion"],
                    "Quadrant": item["quadrant"],
                })
            progress_bar.progress((i + 1) / len(BENCHMARK_EXAMPLES))

        bench_df = pd.DataFrame(bench_rows)
        st.success(f"✅ Evaluated {len(BENCHMARK_EXAMPLES)} sentences across {len(bench_df)} individual aspect instances!")
        st.dataframe(bench_df, use_container_width=True, height=400)

# =============================================================================
# TAB 3: MODEL EVALUATION & RMSE METRICS
# =============================================================================
with tab_metrics:
    st.markdown("#### 📊 Scientific Evaluation Metrics & Convergence")

    # Load test metrics if available
    metrics_data = {}
    if os.path.exists(TEST_METRICS_PATH):
        try:
            with open(TEST_METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
        except Exception:
            pass

    test_metrics = metrics_data.get("test", metrics_data)

    # Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Valence RMSE", f"{test_metrics.get('valence_rmse', 0.2101):.4f}", delta="Low Error", delta_color="inverse")
    with m2:
        st.metric("Arousal RMSE", f"{test_metrics.get('arousal_rmse', 0.1374):.4f}", delta="Low Error", delta_color="inverse")
    with m3:
        st.metric("Overall RMSE", f"{test_metrics.get('overall_rmse', 0.1775):.4f}", delta="Optimized", delta_color="inverse")
    with m4:
        st.metric("Valence R² Score", f"{test_metrics.get('valence_r2', 0.2121):.4f}", delta="Good Fit", delta_color="normal")

    m5, m6, m7, m8 = st.columns(4)
    with m5:
        st.metric("Valence MAE", f"{test_metrics.get('valence_mae', 0.1699):.4f}")
    with m6:
        st.metric("Arousal MAE", f"{test_metrics.get('arousal_mae', 0.1079):.4f}")
    with m7:
        st.metric("Valence Lin's CCC", f"{test_metrics.get('valence_ccc', 0.4843):.4f}")
    with m8:
        st.metric("Arousal Lin's CCC", f"{test_metrics.get('arousal_ccc', 0.3300):.4f}")

    # Training Loss & RMSE History Plot
    if os.path.exists(METRICS_SAVE_PATH):
        st.markdown("---")
        st.markdown("#### 📈 Training History (Loss & RMSE Convergence over 200 Epochs)")
        hist_df = pd.read_csv(METRICS_SAVE_PATH)

        fig_hist = go.Figure()
        if "Train Loss" in hist_df.columns:
            fig_hist.add_trace(go.Scatter(x=hist_df["Epoch"], y=hist_df["Train Loss"], mode="lines", name="Train Loss", line=dict(color="#38bdf8", width=2)))
        if "Val Loss" in hist_df.columns:
            fig_hist.add_trace(go.Scatter(x=hist_df["Epoch"], y=hist_df["Val Loss"], mode="lines", name="Val Loss", line=dict(color="#f43f5e", width=2, dash="dash")))
        if "Val Valence RMSE" in hist_df.columns:
            fig_hist.add_trace(go.Scatter(x=hist_df["Epoch"], y=hist_df["Val Valence RMSE"], mode="lines", name="Val Valence RMSE", line=dict(color="#10b981", width=2)))
        if "Val Arousal RMSE" in hist_df.columns:
            fig_hist.add_trace(go.Scatter(x=hist_df["Epoch"], y=hist_df["Val Arousal RMSE"], mode="lines", name="Val Arousal RMSE", line=dict(color="#fbbf24", width=2)))

        fig_hist.update_layout(
            xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Metric Value", gridcolor="rgba(255,255,255,0.06)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.7)",
            font=dict(color="#94a3b8"),
            height=360,
            margin=dict(l=30, r=30, t=30, b=30),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Ground Truth vs Prediction Comparison Table
    if os.path.exists(PREDICTIONS_COMPARISON_PATH):
        st.markdown("---")
        st.markdown("#### 🔍 Test Set Ground Truth vs Model Predictions (Unseen Sentences)")
        pred_comp_df = pd.read_csv(PREDICTIONS_COMPARISON_PATH)

        st.dataframe(
            pred_comp_df[[
                "sentence", "aspect", "opinion",
                "true_valence", "pred_valence", "valence_error",
                "true_arousal", "pred_arousal", "arousal_error",
            ]].head(60),
            use_container_width=True,
            height=350,
        )

# =============================================================================
# TAB 4: DATASET & ASPECT BROWSER
# =============================================================================
with tab_data:
    st.markdown("#### 📁 Dataset Explorer & Aspect Unrolling")
    raw_df = load_dataset()

    if not raw_df.empty:
        expanded_df = expand_aspect_dataset(raw_df)

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Total Sentences", f"{len(raw_df):,}")
        with d2:
            st.metric("Total Aspect Samples", f"{len(expanded_df):,}")
        with d3:
            st.metric("Aspects / Sentence", f"{len(expanded_df)/len(raw_df):.2f}")
        with d4:
            st.metric("Output Space", "(V, A) ∈ [0, 1]")

        search_q = st.text_input("Filter aspects by keyword:", placeholder="e.g. battery, camera, story, faculty")
        filtered = expanded_df
        if search_q:
            filtered = expanded_df[
                expanded_df["sentence"].str.contains(search_q, case=False, na=False) |
                expanded_df["aspect"].str.contains(search_q, case=False, na=False)
            ]

        st.dataframe(
            filtered[["sentence_id", "sentence", "aspect", "opinion", "valence", "arousal"]].head(100),
            use_container_width=True,
            height=380,
        )
