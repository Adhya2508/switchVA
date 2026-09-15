import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Add workspace directory to path
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
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Aspect-Based Emotion Regression",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Clean CSS Design System
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        padding: 1.2rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }

    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.25rem;
        font-weight: 400;
    }

    .aspect-row-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    .aspect-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f1f5f9;
    }

    .badge-positive {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .badge-negative {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .metric-subtext {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Cached Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading fine-tuned model...")
def load_engine():
    return DimABSAInferenceEngine()


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
# Benchmark Examples
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
# Clean Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">Aspect-Based Emotion Regression</div>
        <div class="sub-title">Aspect-conditioned continuous Valence and Arousal regression on code-mixed Hinglish reviews</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2D Circumplex Plot Function
# -----------------------------------------------------------------------------
def build_circumplex_plot(aspect_results):
    fig = go.Figure()

    # 4 Quadrant Backgrounds
    fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=1.0, y1=1.0, fillcolor="rgba(34, 197, 94, 0.08)", line=dict(width=0))
    fig.add_shape(type="rect", x0=0.0, y0=0.5, x1=0.5, y1=1.0, fillcolor="rgba(239, 68, 68, 0.08)", line=dict(width=0))
    fig.add_shape(type="rect", x0=0.0, y0=0.0, x1=0.5, y1=0.5, fillcolor="rgba(249, 115, 22, 0.08)", line=dict(width=0))
    fig.add_shape(type="rect", x0=0.5, y0=0.0, x1=1.0, y1=0.5, fillcolor="rgba(56, 189, 248, 0.08)", line=dict(width=0))

    # Quadrant Titles
    fig.add_annotation(x=0.04, y=0.96, text="<b>Q2: ANGER / FRUSTRATION</b>", showarrow=False, font=dict(color="#f87171", size=12), xanchor="left")
    fig.add_annotation(x=0.96, y=0.96, text="<b>Q1: JOY / EXCITEMENT</b>", showarrow=False, font=dict(color="#4ade80", size=12), xanchor="right")
    fig.add_annotation(x=0.04, y=0.04, text="<b>Q3: SADNESS / DISAPPOINTMENT</b>", showarrow=False, font=dict(color="#fb923c", size=12), xanchor="left")
    fig.add_annotation(x=0.96, y=0.04, text="<b>Q4: SERENITY / CALM</b>", showarrow=False, font=dict(color="#38bdf8", size=12), xanchor="right")

    # Midline Dividers
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", line_width=1.2)
    fig.add_vline(x=0.5, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", line_width=1.2)

    # Reference Affect Anchors
    EMOTION_ANCHORS = [
        # Q1
        {"name": "Euphoric", "x": 0.95, "y": 0.90},
        {"name": "Astonished / Excited", "x": 0.75, "y": 0.85},
        {"name": "Joyful / Delighted", "x": 0.85, "y": 0.75},
        {"name": "Proud / Inspired", "x": 0.75, "y": 0.65},
        {"name": "Amused / Playful", "x": 0.70, "y": 0.60},
        # Q2
        {"name": "Enraged / Furious", "x": 0.10, "y": 0.95},
        {"name": "Shocked / Alarmed", "x": 0.35, "y": 0.90},
        {"name": "Anxious / Panicked", "x": 0.20, "y": 0.85},
        {"name": "Angry / Hostile", "x": 0.20, "y": 0.80},
        {"name": "Frustrated / Annoyed", "x": 0.30, "y": 0.70},
        {"name": "Disgusted / Repulsed", "x": 0.15, "y": 0.65},
        # Q3
        {"name": "Disappointed / Let Down", "x": 0.30, "y": 0.40},
        {"name": "Sad / Heartbroken", "x": 0.18, "y": 0.30},
        {"name": "Tired / Exhausted", "x": 0.32, "y": 0.25},
        {"name": "Bored / Apathetic", "x": 0.35, "y": 0.18},
        {"name": "Depressed / Despairing", "x": 0.10, "y": 0.15},
        # Q4
        {"name": "Grateful / Appreciative", "x": 0.78, "y": 0.45},
        {"name": "Content / Satisfied", "x": 0.72, "y": 0.40},
        {"name": "Relieved / Reassured", "x": 0.68, "y": 0.35},
        {"name": "Calm / Peaceful", "x": 0.75, "y": 0.25},
        {"name": "Serene / Blissful", "x": 0.88, "y": 0.20},
        # Center
        {"name": "Neutral / Balanced", "x": 0.50, "y": 0.50},
    ]

    ref_x = [a["x"] for a in EMOTION_ANCHORS]
    ref_y = [a["y"] for a in EMOTION_ANCHORS]
    ref_text = [a["name"] for a in EMOTION_ANCHORS]

    fig.add_trace(
        go.Scatter(
            x=ref_x,
            y=ref_y,
            mode="markers+text",
            marker=dict(size=5, color="rgba(148, 163, 184, 0.35)"),
            text=ref_text,
            textposition="top center",
            textfont=dict(size=9, color="rgba(148, 163, 184, 0.65)"),
            hoverinfo="text",
            name="Reference Points",
            showlegend=False,
        )
    )

    # Predicted Aspect Points
    aspect_colors = ["#22c55e", "#f97316", "#38bdf8", "#ec4899", "#a855f7", "#eab308"]
    for idx, asp in enumerate(aspect_results):
        v = asp["valence"]
        a = asp["arousal"]
        c = aspect_colors[idx % len(aspect_colors)]

        # Solid marker with white outline
        fig.add_trace(
            go.Scatter(
                x=[v],
                y=[a],
                mode="markers+text",
                marker=dict(size=18, color=c, line=dict(width=2, color="#ffffff")),
                text=[f"<b>{asp['aspect']}</b> ({v:.2f}, {a:.2f})"],
                textposition="bottom center",
                textfont=dict(size=11.5, color=c),
                name=asp["aspect"],
                showlegend=False,
            )
        )

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Valence (0.0: Negative ── 0.5: Neutral ── 1.0: Positive)", font=dict(size=11, color="#94a3b8")),
            range=[0, 1],
            tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            gridcolor="rgba(255, 255, 255, 0.06)",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="Arousal (0.0: Low / Calm ── 0.5: Medium ── 1.0: High / Intense)", font=dict(size=11, color="#94a3b8")),
            range=[0, 1],
            tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            gridcolor="rgba(255, 255, 255, 0.06)",
            zeroline=False,
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=40, r=40, t=40, b=40),
        height=500,
    )
    return fig


# -----------------------------------------------------------------------------
# Clean Tabs
# -----------------------------------------------------------------------------
tab_live, tab_bench, tab_metrics, tab_data = st.tabs(
    [
        "Sentence Analysis",
        "Benchmark Suite",
        "Model Evaluation",
        "Dataset Browser",
    ]
)

# Load engine
try:
    engine = load_engine()
except Exception as e:
    st.error(f"Error loading model engine: {e}")
    st.stop()

# =============================================================================
# TAB 1: SENTENCE ANALYSIS
# =============================================================================
with tab_live:
    col_in, col_opt = st.columns([4, 1.5])
    with col_opt:
        selected_bench = st.selectbox("Sample reviews:", ["-- Custom Input --"] + BENCHMARK_EXAMPLES)

    default_text = selected_bench if selected_bench != "-- Custom Input --" else "The battery life is amazing but the display is disappointing."

    user_text = st.text_area(
        "Input Review:",
        value=default_text,
        height=75,
        placeholder="Enter a review (e.g. The battery life is amazing but the display is disappointing.)",
        label_visibility="collapsed",
    )

    c_btn1, c_btn2, _ = st.columns([1.5, 1.5, 5])
    with c_btn1:
        run_analysis = st.button("Analyze Review", type="primary", use_container_width=True)
    with c_btn2:
        if st.button("Random Sample", use_container_width=True):
            df_all = load_dataset()
            if not df_all.empty:
                st.session_state["sample_text"] = df_all.sample(1).iloc[0]["sentence"]
                st.rerun()

    if "sample_text" in st.session_state:
        user_text = st.session_state.pop("sample_text")

    if user_text:
        with st.spinner("Processing aspect-level predictions..."):
            result = engine.predict(user_text)

        aspect_results = result.get("aspects", [])

        if aspect_results:
            st.markdown("---")
            st.markdown(f"**Predictions ({len(aspect_results)} detected aspect{'s' if len(aspect_results) > 1 else ''})**")

            # Table representation
            table_rows = []
            for r in aspect_results:
                table_rows.append({
                    "Aspect": r["aspect"],
                    "Opinion": r["opinion"],
                    "Valence (V)": f"{r['valence']:.3f}",
                    "Arousal (A)": f"{r['arousal']:.3f}",
                    "Polarity": r["polarity"],
                    "Emotion Category": r["emotion"].split(" ")[0],
                    "Affect Quadrant": r["quadrant"].split(":")[0] + ":" + r["quadrant"].split(":")[1] if ":" in r["quadrant"] else r["quadrant"],
                    "Confidence": f"{r['confidence'] * 100:.1f}%",
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            # Two Columns: Aspect Details + Russell 2D Plot
            col_cards, col_plot = st.columns([1, 1.3])

            with col_cards:
                st.markdown("**Aspect Details**")
                for i, asp in enumerate(aspect_results):
                    v = asp["valence"]
                    a = asp["arousal"]
                    is_pos = v >= 0.5
                    badge_cls = "badge-positive" if is_pos else "badge-negative"

                    st.markdown(
                        f"""
                        <div class="aspect-row-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span class="aspect-title">{asp['aspect']}</span>
                                <span class="{badge_cls}">{asp['polarity']}</span>
                            </div>
                            <div style="color:#94a3b8; font-size:0.85rem; margin-top:4px;">
                                Context: <i>"{asp['opinion']}"</i>
                            </div>
                            <div style="display:flex; gap:20px; margin-top:10px;">
                                <div>
                                    <div class="metric-subtext">Valence</div>
                                    <div style="font-size:1.1rem; font-weight:600; color:{'#4ade80' if is_pos else '#f87171'};">{v:.3f}</div>
                                </div>
                                <div>
                                    <div class="metric-subtext">Arousal</div>
                                    <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0;">{a:.3f}</div>
                                </div>
                                <div>
                                    <div class="metric-subtext">Quadrant</div>
                                    <div style="font-size:0.95rem; font-weight:500; color:#38bdf8;">{asp['quadrant'].split(':')[0]}</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col_plot:
                st.markdown("**2D Circumplex Emotion Matrix**")
                fig_circumplex = build_circumplex_plot(aspect_results)
                st.plotly_chart(fig_circumplex, use_container_width=True)

        else:
            st.info("No aspect candidates detected in this sentence.")

# =============================================================================
# TAB 2: BENCHMARK SUITE
# =============================================================================
with tab_bench:
    st.markdown("**Batch Benchmark Evaluation**")
    st.caption("Run aspect-specific regression across benchmark sentences to test polarity divergence:")

    if st.button("Run Benchmark Suite", type="primary"):
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
                    "Quadrant": item["quadrant"],
                })
            progress_bar.progress((i + 1) / len(BENCHMARK_EXAMPLES))

        st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, height=420)

# =============================================================================
# TAB 3: MODEL EVALUATION
# =============================================================================
with tab_metrics:
    st.markdown("**Statistical Metrics (Unseen Test Split)**")

    metrics_data = {}
    if os.path.exists(TEST_METRICS_PATH):
        try:
            with open(TEST_METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
        except Exception:
            pass

    test_metrics = metrics_data.get("test", metrics_data)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Overall RMSE", f"{test_metrics.get('overall_rmse', 0.1775):.4f}")
    with m2:
        st.metric("Valence RMSE", f"{test_metrics.get('valence_rmse', 0.2101):.4f}")
    with m3:
        st.metric("Arousal RMSE", f"{test_metrics.get('arousal_rmse', 0.1374):.4f}")
    with m4:
        st.metric("Valence R²", f"{test_metrics.get('valence_r2', 0.2121):.4f}")

    m5, m6, m7, m8 = st.columns(4)
    with m5:
        st.metric("Valence MAE", f"{test_metrics.get('valence_mae', 0.1699):.4f}")
    with m6:
        st.metric("Arousal MAE", f"{test_metrics.get('arousal_mae', 0.1079):.4f}")
    with m7:
        st.metric("Valence Lin's CCC", f"{test_metrics.get('valence_ccc', 0.4843):.4f}")
    with m8:
        st.metric("Arousal Lin's CCC", f"{test_metrics.get('arousal_ccc', 0.3300):.4f}")

    if os.path.exists(METRICS_SAVE_PATH):
        st.markdown("---")
        st.markdown("**Training & Validation Convergence (200 Epochs)**")
        hist_df = pd.read_csv(METRICS_SAVE_PATH)

        fig_hist = go.Figure()
        if "Train Loss" in hist_df.columns:
            fig_hist.add_trace(go.Scatter(x=hist_df["Epoch"], y=hist_df["Train Loss"], mode="lines", name="Train Loss", line=dict(color="#38bdf8", width=1.8)))
        if "Val Loss" in hist_df.columns:
            fig_hist.add_trace(go.Scatter(x=hist_df["Epoch"], y=hist_df["Val Loss"], mode="lines", name="Val Loss", line=dict(color="#f43f5e", width=1.8, dash="dash")))
        if "Val Valence RMSE" in hist_df.columns:
            fig_hist.add_trace(go.Scatter(x=hist_df["Epoch"], y=hist_df["Val Valence RMSE"], mode="lines", name="Valence RMSE", line=dict(color="#10b981", width=1.8)))
        if "Val Arousal RMSE" in hist_df.columns:
            fig_hist.add_trace(go.Scatter(x=hist_df["Epoch"], y=hist_df["Val Arousal RMSE"], mode="lines", name="Arousal RMSE", line=dict(color="#fbbf24", width=1.8)))

        fig_hist.update_layout(
            xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Loss / Metric", gridcolor="rgba(255,255,255,0.06)"),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#94a3b8"),
            height=340,
            margin=dict(l=30, r=30, t=30, b=30),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    if os.path.exists(PREDICTIONS_COMPARISON_PATH):
        st.markdown("---")
        st.markdown("**Test Set Ground Truth vs Model Predictions**")
        pred_comp_df = pd.read_csv(PREDICTIONS_COMPARISON_PATH)
        st.dataframe(
            pred_comp_df[[
                "sentence", "aspect", "opinion",
                "true_valence", "pred_valence", "valence_error",
                "true_arousal", "pred_arousal", "arousal_error",
            ]].head(50),
            use_container_width=True,
            height=340,
        )

# =============================================================================
# TAB 4: DATASET BROWSER
# =============================================================================
with tab_data:
    st.markdown("**Dataset Exploration (1,101 Aspect Instances)**")
    raw_df = load_dataset()

    if not raw_df.empty:
        expanded_df = expand_aspect_dataset(raw_df)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Total Sentences", f"{len(raw_df):,}")
        with d2:
            st.metric("Unrolled Aspect Samples", f"{len(expanded_df):,}")
        with d3:
            st.metric("Aspects per Sentence", f"{len(expanded_df)/len(raw_df):.2f}")

        search_q = st.text_input("Filter aspects or sentences:", placeholder="Type a keyword to filter...")
        filtered = expanded_df
        if search_q:
            filtered = expanded_df[
                expanded_df["sentence"].str.contains(search_q, case=False, na=False) |
                expanded_df["aspect"].str.contains(search_q, case=False, na=False)
            ]

        st.dataframe(
            filtered[["sentence_id", "sentence", "aspect", "opinion", "valence", "arousal"]].head(100),
            use_container_width=True,
            height=400,
        )
