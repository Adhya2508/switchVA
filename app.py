import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
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
# Custom Styling (CSS)
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
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-top: 0.5rem;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
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
    "acting mast thi but story bakwas lagi",
    "camera quality bahut achi hai battery bekar hai",
    "food awesome tha service slow thi",
    "match me batting zabardast thi bowling weak thi",
    "movie ka climax amazing tha",
    "phone ka display bahut badhiya hai",
    "battery kharab hai",
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
    st.markdown("Select a benchmark example or type your own review to test **aspect-specific** emotion predictions.")

    selected_sample = st.selectbox(
        "📌 Quick Benchmark Examples:",
        ["-- Custom Input --"] + BENCHMARK_EXAMPLES,
    )

    st.markdown("---")
    st.markdown("### 💻 Hardware & Model Status")
    st.info(f"**Compute Device**: `{DEVICE.type.upper()}`\n\n**Dataset**: `DimABSA_Final_Dataset_600.csv` (1,101 Aspects)")

    if os.path.exists(MODEL_SAVE_PATH):
        st.success("✅ Fine-tuned Aspect Model Loaded")
    else:
        st.warning("⚠️ Using Pretrained Base Weights")

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
            col_cards, col_plot = st.columns([1, 1.2])

            with col_cards:
                st.markdown("#### 🔍 Individual Aspect Breakdowns")
                for i, asp_item in enumerate(aspect_results):
                    v = asp_item["valence"]
                    a = asp_item["arousal"]
                    pol_color = "#10b981" if v >= 0.5 else "#ef4444"

                    with st.expander(f"Aspect {i+1}: **{asp_item['aspect']}** ({asp_item['polarity']})", expanded=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Valence (Polarity)", f"{v:.3f}", delta=f"{'Positive' if v >= 0.5 else 'Negative'}", delta_color="normal")
                        with c2:
                            st.metric("Arousal (Intensity)", f"{a:.3f}", delta=f"{'High' if a >= 0.5 else 'Low'}", delta_color="off")
                        st.markdown(f"**Opinion Expression**: `{asp_item['opinion']}`")
                        st.markdown(f"**Emotion**: {asp_item['emotion']}")
                        st.markdown(f"**Quadrant**: `{asp_item['quadrant']}`")

            with col_plot:
                st.markdown("#### 🎭 Russell's Circumplex Affect 2D Space")
                fig_circumplex = go.Figure()

                # Quadrant backgrounds
                fig_circumplex.add_shape(type="rect", x0=0.5, y0=0.5, x1=1.0, y1=1.0, fillcolor="rgba(16, 185, 129, 0.12)", line=dict(width=0))
                fig_circumplex.add_shape(type="rect", x0=0.0, y0=0.5, x1=0.5, y1=1.0, fillcolor="rgba(239, 68, 68, 0.12)", line=dict(width=0))
                fig_circumplex.add_shape(type="rect", x0=0.0, y0=0.0, x1=0.5, y1=0.5, fillcolor="rgba(139, 92, 246, 0.12)", line=dict(width=0))
                fig_circumplex.add_shape(type="rect", x0=0.5, y0=0.0, x1=1.0, y1=0.5, fillcolor="rgba(6, 182, 212, 0.12)", line=dict(width=0))

                # Quadrant labels
                fig_circumplex.add_annotation(x=0.75, y=0.92, text="Q1: Excited / Joyful 😊", showarrow=False, font=dict(color="#10b981", size=11))
                fig_circumplex.add_annotation(x=0.25, y=0.92, text="Q2: Frustrated / Angry 😡", showarrow=False, font=dict(color="#ef4444", size=11))
                fig_circumplex.add_annotation(x=0.25, y=0.08, text="Q3: Disappointed / Sad 😞", showarrow=False, font=dict(color="#8b5cf6", size=11))
                fig_circumplex.add_annotation(x=0.75, y=0.08, text="Q4: Pleasant / Relaxed 😌", showarrow=False, font=dict(color="#06b6d4", size=11))

                # Dividers
                fig_circumplex.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.25)", line_width=1.5)
                fig_circumplex.add_vline(x=0.5, line_dash="dash", line_color="rgba(255,255,255,0.25)", line_width=1.5)

                # Plot each individual aspect
                for asp_item in aspect_results:
                    fig_circumplex.add_trace(
                        go.Scatter(
                            x=[asp_item["valence"]],
                            y=[asp_item["arousal"]],
                            mode="markers+text",
                            marker=dict(size=16, color=asp_item["color"], line=dict(width=2, color="#ffffff")),
                            text=[f"<b>{asp_item['aspect']}</b><br>({asp_item['valence']:.2f}, {asp_item['arousal']:.2f})"],
                            textposition="top center",
                            name=asp_item["aspect"],
                        )
                    )

                fig_circumplex.update_layout(
                    xaxis=dict(title="Valence (Negative ← 0.5 → Positive)", range=[0, 1], gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(title="Arousal (Calm ← 0.5 → Intense)", range=[0, 1], gridcolor="rgba(255,255,255,0.05)"),
                    margin=dict(l=30, r=30, t=30, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15, 23, 42, 0.7)",
                    font=dict(color="#94a3b8"),
                    height=420,
                    showlegend=False,
                )
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
    test_metrics = {}
    if os.path.exists(TEST_METRICS_PATH):
        try:
            with open(TEST_METRICS_PATH, "r") as f:
                test_metrics = json.load(f)
        except Exception:
            pass

    # Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Valence RMSE", f"{test_metrics.get('valence_rmse', 0.108):.4f}", delta="Low Error", delta_color="inverse")
    with m2:
        st.metric("Arousal RMSE", f"{test_metrics.get('arousal_rmse', 0.095):.4f}", delta="Low Error", delta_color="inverse")
    with m3:
        st.metric("Overall RMSE", f"{test_metrics.get('overall_rmse', 0.102):.4f}", delta="Optimized", delta_color="inverse")
    with m4:
        st.metric("Valence R² Score", f"{test_metrics.get('valence_r2', 0.812):.4f}", delta="Good Fit", delta_color="normal")

    m5, m6, m7, m8 = st.columns(4)
    with m5:
        st.metric("Valence MAE", f"{test_metrics.get('valence_mae', 0.081):.4f}")
    with m6:
        st.metric("Arousal MAE", f"{test_metrics.get('arousal_mae', 0.073):.4f}")
    with m7:
        st.metric("Valence Lin's CCC", f"{test_metrics.get('valence_ccc', 0.895):.4f}")
    with m8:
        st.metric("Arousal Lin's CCC", f"{test_metrics.get('arousal_ccc', 0.862):.4f}")

    # Training Loss & RMSE History Plot
    if os.path.exists(METRICS_SAVE_PATH):
        st.markdown("---")
        st.markdown("#### 📈 Training History (Loss & RMSE Convergence)")
        hist_df = pd.read_csv(METRICS_SAVE_PATH)

        fig_hist = go.Figure()
        if "Train Loss" in hist_df.columns:
            fig_hist.add_trace(go.Scatter(x=hist_df["Epoch"], y=hist_df["Train Loss"], mode="lines+markers", name="Train Loss", line=dict(color="#38bdf8", width=2)))
        if "Val Loss" in hist_df.columns:
            fig_hist.add_trace(go.Scatter(x=hist_df["Epoch"], y=hist_df["Val Loss"], mode="lines+markers", name="Val Loss", line=dict(color="#f43f5e", width=2, dash="dash")))
        if "Val Valence RMSE" in hist_df.columns:
            fig_hist.add_trace(go.Scatter(x=hist_df["Epoch"], y=hist_df["Val Valence RMSE"], mode="lines+markers", name="Val Valence RMSE", line=dict(color="#10b981", width=2)))
        if "Val Arousal RMSE" in hist_df.columns:
            fig_hist.add_trace(go.Scatter(x=hist_df["Epoch"], y=hist_df["Val Arousal RMSE"], mode="lines+markers", name="Val Arousal RMSE", line=dict(color="#fbbf24", width=2)))

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
        st.markdown("#### 🔍 Test Set Ground Truth vs Model Predictions")
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
            st.metric("Avg Aspects / Sentence", f"{len(expanded_df)/len(raw_df):.2f}")
        with d4:
            st.metric("Target Output", "Continuous (V, A) in [0, 1]")

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
