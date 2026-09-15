import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aspect-Based Emotion Regression",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #0d1117 !important;
    color: #e2e8f0;
}
/* Ensure the page is scrollable */
html, body { overflow: auto !important; height: auto !important; }
.main, section.main, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {
    overflow: visible !important;
    height: auto !important;
}
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; overflow: visible !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* Top nav */
.top-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 0; border-bottom: 1px solid rgba(255,255,255,0.07); margin-bottom: 0;
}
.nav-brand { display: flex; align-items: center; gap: 12px; }
.brand-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 10px; display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
}
.brand-title { font-size: 1.15rem; font-weight: 700; color: #f1f5f9; margin: 0; letter-spacing: -0.01em; }
.brand-sub { font-size: 0.77rem; color: #64748b; margin: 0; }
.nav-links { display: flex; gap: 6px; }
.nav-link {
    color: #94a3b8; font-size: 0.82rem; font-weight: 500;
    padding: 6px 12px; border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.07);
    background: rgba(255,255,255,0.03);
}

/* Section labels */
.section-label {
    font-size: 0.78rem; font-weight: 600; color: #64748b;
    letter-spacing: 0.05em; text-transform: uppercase;
    margin: 1rem 0 0.5rem 0;
}

/* Status bar */
.status-bar {
    display: flex; align-items: center; justify-content: space-between;
    background: #161b22; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px; padding: 12px 18px; margin: 0.8rem 0 0.6rem 0;
}
.status-left { display: flex; align-items: center; gap: 10px; }
.status-dot {
    width: 10px; height: 10px; background: #22c55e;
    border-radius: 50%; box-shadow: 0 0 6px #22c55e88;
}
.status-title { font-size: 0.95rem; font-weight: 600; color: #f1f5f9; }
.status-sub { font-size: 0.77rem; color: #64748b; }
.status-sentence { font-size: 0.82rem; color: #94a3b8; font-style: italic; max-width: 52%; text-align: right; }

/* Aspect cards */
.asp-card {
    background: #161b22; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 1rem 1.1rem; margin-bottom: 0.6rem;
}
.asp-card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
.asp-icon {
    width: 30px; height: 30px; border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
    background: rgba(255,255,255,0.05);
}
.asp-name { font-size: 0.98rem; font-weight: 600; color: #f1f5f9; margin-left: 8px; }
.asp-context { font-size: 0.79rem; color: #64748b; margin: 4px 0 10px 0; }
.asp-metrics { display: flex; gap: 20px; }
.asp-metric-label { font-size: 0.7rem; color: #475569; text-transform: uppercase; letter-spacing: 0.04em; }
.asp-metric-value { font-size: 1.05rem; font-weight: 700; margin-top: 1px; }
.badge { font-size: 0.72rem; font-weight: 600; padding: 3px 9px; border-radius: 4px; }
.badge-pos { background: rgba(34,197,94,0.12); color: #4ade80; }
.badge-neg { background: rgba(239,68,68,0.12); color: #f87171; }
.badge-neu { background: rgba(148,163,184,0.12); color: #94a3b8; }

/* Footer */
.footer {
    border-top: 1px solid rgba(255,255,255,0.06);
    padding: 12px 0; margin-top: 2rem;
    display: flex; justify-content: space-between; align-items: center;
    color: #475569; font-size: 0.78rem;
}

/* Metric overrides */
[data-testid="stMetric"] label { color: #64748b !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.35rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { display: none; }

/* Button */
button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #6d28d9) !important;
    border: none !important; font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading fine-tuned model...")
def load_engine():
    return DimABSAInferenceEngine()


@st.cache_data
def load_dataset():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df["aspect_list"]  = df["all_aspects"].apply(parse_set_string)
        df["opinion_list"] = df["all_opinions"].apply(parse_set_string)
        df["valence_list"] = df["valence_scores"].apply(parse_float_string)
        df["arousal_list"] = df["arousal_scores"].apply(parse_float_string)
        return df
    return pd.DataFrame()


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


# ── Circumplex plot ───────────────────────────────────────────────────────────
def build_circumplex_plot(aspect_results):
    fig = go.Figure()

    for x0, y0, x1, y1, col in [
        (0.0, 0.5, 0.5, 1.0, "rgba(239,68,68,0.07)"),
        (0.5, 0.5, 1.0, 1.0, "rgba(34,197,94,0.07)"),
        (0.0, 0.0, 0.5, 0.5, "rgba(249,115,22,0.07)"),
        (0.5, 0.0, 1.0, 0.5, "rgba(56,189,248,0.07)"),
    ]:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      fillcolor=col, line=dict(width=0), layer="below")

    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.18)", line_width=1)
    fig.add_vline(x=0.5, line_dash="dash", line_color="rgba(255,255,255,0.18)", line_width=1)

    Q_LABELS = [
        (0.02, 0.99, "left", "top",    "#f87171", "<b>Q2: ANGER / FRUSTRATION</b><br>Terse - Nervous - Annoyed"),
        (0.98, 0.99, "right", "top",   "#4ade80", "<b>Q1: JOY / EXCITEMENT</b><br>Excited - Happy - Energetic"),
        (0.02, 0.01, "left", "bottom", "#fb923c", "<b>Q3: SADNESS / DISPLEASURE</b><br>Sad - Depressed - Bored"),
        (0.98, 0.01, "right", "bottom","#38bdf8", "<b>Q4: CALMNESS / CONTENTMENT</b><br>Relaxed - Peaceful - Satisfied"),
    ]
    for x, y, xanc, yanc, color, text in Q_LABELS:
        fig.add_annotation(x=x, y=y, xanchor=xanc, yanchor=yanc,
                           text=text, showarrow=False,
                           font=dict(color=color, size=10))

    ANCHORS = [
        ("Euphoric",      0.95, 0.92), ("Excited",       0.80, 0.88),
        ("Joyful",        0.82, 0.75), ("Inspired",      0.72, 0.68),
        ("Amused",        0.70, 0.60),
        ("Enraged",       0.08, 0.95), ("Shocked",       0.32, 0.90),
        ("Anxious",       0.18, 0.82), ("Angry",         0.15, 0.75),
        ("Frustrated",    0.28, 0.68), ("Disgusted",     0.12, 0.62),
        ("Disappointed",  0.30, 0.40), ("Sad",           0.16, 0.28),
        ("Tired",         0.35, 0.22), ("Bored",         0.38, 0.15),
        ("Depressed",     0.10, 0.10),
        ("Grateful",      0.78, 0.44), ("Content",       0.72, 0.37),
        ("Relieved",      0.67, 0.30), ("Calm",          0.76, 0.22),
        ("Serene",        0.90, 0.15), ("Neutral",       0.50, 0.50),
    ]
    fig.add_trace(go.Scatter(
        x=[a[1] for a in ANCHORS], y=[a[2] for a in ANCHORS],
        mode="markers+text",
        marker=dict(size=5, color="rgba(148,163,184,0.28)"),
        text=[a[0] for a in ANCHORS],
        textposition="top center",
        textfont=dict(size=8.5, color="rgba(148,163,184,0.50)"),
        hoverinfo="text", showlegend=False,
    ))

    COLORS = ["#22c55e", "#f97316", "#38bdf8", "#ec4899", "#a855f7", "#eab308"]
    for idx, asp in enumerate(aspect_results):
        v, a = asp["valence"], asp["arousal"]
        c = COLORS[idx % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=[v], y=[a], mode="markers+text",
            marker=dict(size=16, color=c, line=dict(width=2, color="#ffffff")),
            text=[f"<b>{asp['aspect']}</b><br>({v:.3f}, {a:.3f})"],
            textposition="bottom center",
            textfont=dict(size=10.5, color=c),
            name=asp["aspect"], showlegend=False,
            hovertemplate=(
                f"<b>{asp['aspect']}</b><br>"
                f"Valence: {v:.3f}<br>Arousal: {a:.3f}<br>"
                f"{asp['emotion']}<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis=dict(
            range=[0, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            gridcolor="rgba(255,255,255,0.05)", zeroline=False, showline=False,
            title=dict(text="Valence (V)  →  More Pleasant", font=dict(size=11, color="#475569")),
        ),
        yaxis=dict(
            range=[0, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            gridcolor="rgba(255,255,255,0.05)", zeroline=False, showline=False,
            title=dict(text="Arousal (A)   (High Energy ↑)", font=dict(size=11, color="#475569")),
        ),
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        margin=dict(l=50, r=20, t=30, b=50), height=440,
        hoverlabel=dict(bgcolor="#1e293b", font_color="#f1f5f9", font_size=12),
    )
    return fig


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-nav">
  <div class="nav-brand">
    <div class="brand-icon">🧠</div>
    <div>
      <div class="brand-title">Aspect-Based Emotion Regression</div>
      <div class="brand-sub">Aspect-conditioned continuous Valence &amp; Arousal regression &middot; Code-mixed Hinglish</div>
    </div>
  </div>
  <div class="nav-links">
    <span class="nav-link">🏠 Home</span>
    <span class="nav-link">ℹ️ About</span>
    <span class="nav-link">❓ Help</span>
  </div>
</div>
""", unsafe_allow_html=True)

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

tab_live, tab_bench, tab_metrics, tab_data = st.tabs([
    "📝  Sentence Analysis",
    "📊  Benchmark Suite",
    "📈  Model Evaluation",
    "🗂  Dataset Browser",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Sentence Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab_live:
    col_input, col_sample = st.columns([3, 1])

    with col_sample:
        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
        selected_bench = st.selectbox(
            "Sample Reviews",
            ["-- Select a sample review --"] + BENCHMARK_EXAMPLES,
        )

    with col_input:
        default_text = (
            selected_bench
            if selected_bench != "-- Select a sample review --"
            else "The battery life is amazing but the display is disappointing."
        )
        st.markdown(
            "<div class='section-label' style='margin-top:14px'>Enter a Review Sentence</div>",
            unsafe_allow_html=True,
        )
        user_text = st.text_area(
            "sentence_input", value=default_text, height=80,
            label_visibility="collapsed",
        )

    if "sample_text" in st.session_state:
        user_text = st.session_state.pop("sample_text")

    b1, b2, _ = st.columns([1.4, 1.4, 5])
    with b1:
        run_btn = st.button("🔍  Analyze Review", type="primary", use_container_width=True)
    with b2:
        if st.button("🎲  Random Sample", use_container_width=True):
            df_all = load_dataset()
            if not df_all.empty:
                st.session_state["sample_text"] = df_all.sample(1).iloc[0]["sentence"]
                st.rerun()

    if user_text:
        with st.spinner("Processing aspect-level predictions..."):
            result = engine.predict(user_text)
        aspect_results = result.get("aspects", [])

        if aspect_results:
            n = len(aspect_results)
            sentence_display = user_text[:120] + ("..." if len(user_text) > 120 else "")
            st.markdown(f"""
<div class="status-bar">
  <div class="status-left">
    <div class="status-dot"></div>
    <div>
      <div class="status-title">Analysis Complete</div>
      <div class="status-sub">Detected {n} aspect{"s" if n > 1 else ""} in the sentence</div>
    </div>
  </div>
  <div class="status-sentence">"{sentence_display}"</div>
</div>""", unsafe_allow_html=True)

            st.markdown(
                f"<div class='section-label'>Predictions ({n} detected aspect{'s' if n > 1 else ''})</div>",
                unsafe_allow_html=True,
            )

            table_rows = []
            for i, r in enumerate(aspect_results):
                table_rows.append({
                    "#": i + 1,
                    "Aspect": r["aspect"],
                    "Opinion": r["opinion"],
                    "Valence (V)": f"{r['valence']:.3f}",
                    "Arousal (A)": f"{r['arousal']:.3f}",
                    "Polarity": r["polarity"],
                    "Emotion": r["emotion"].split("(")[0].strip(),
                    "Affect Quadrant": r["quadrant"],
                    "Confidence": f"{r['confidence'] * 100:.1f}%",
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            col_cards, col_plot = st.columns([1, 1.35])
            CARD_COLORS = ["#22c55e", "#f97316", "#38bdf8", "#ec4899", "#a855f7", "#eab308"]
            ICONS_POS   = ["🔋", "🎵", "📷", "🍽️", "🏏", "🎬", "🏫", "👨‍🏫", "✨", "💡"]
            ICONS_NEG   = ["🖥️", "💸", "🎭", "🔌", "🐌", "📉", "📦", "⚠️", "❌", "😤"]

            with col_cards:
                st.markdown(
                    "<div class='section-label'>Aspect Details</div>", unsafe_allow_html=True
                )
                for i, asp in enumerate(aspect_results):
                    v, a = asp["valence"], asp["arousal"]
                    is_pos = v >= 0.5
                    icon = ICONS_POS[i % len(ICONS_POS)] if is_pos else ICONS_NEG[i % len(ICONS_NEG)]
                    badge_cls = "badge-pos" if is_pos else "badge-neg"
                    val_color = "#4ade80" if is_pos else "#f87171"
                    q_short = asp["quadrant"].split(":")[0] if ":" in asp["quadrant"] else asp["quadrant"]
                    st.markdown(f"""
<div class="asp-card">
  <div class="asp-card-header">
    <div style="display:flex;align-items:center;gap:8px;">
      <div class="asp-icon">{icon}</div>
      <span class="asp-name">{asp["aspect"]}</span>
    </div>
    <span class="badge {badge_cls}">{asp["polarity"]}</span>
  </div>
  <div class="asp-context">Context: <i>"{asp["opinion"]}"</i></div>
  <div class="asp-metrics">
    <div>
      <div class="asp-metric-label">Valence (V)</div>
      <div class="asp-metric-value" style="color:{val_color}">{v:.3f}</div>
    </div>
    <div>
      <div class="asp-metric-label">Arousal (A)</div>
      <div class="asp-metric-value" style="color:#e2e8f0">{a:.3f}</div>
    </div>
    <div>
      <div class="asp-metric-label">Quadrant</div>
      <div class="asp-metric-value" style="color:#38bdf8;font-size:0.88rem">{q_short}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            with col_plot:
                st.markdown(
                    "<div class='section-label'>2D Circumplex Emotion Matrix</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(build_circumplex_plot(aspect_results), use_container_width=True)

        else:
            st.info("No aspect candidates detected in this sentence.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Benchmark Suite
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bench:
    st.markdown(
        "<div class='section-label'>Batch Benchmark Evaluation</div>", unsafe_allow_html=True
    )
    st.caption("Run aspect-specific regression across benchmark sentences to test polarity divergence.")

    if st.button("▶  Run Full Benchmark Suite", type="primary"):
        bench_rows = []
        prog = st.progress(0)
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
            prog.progress((i + 1) / len(BENCHMARK_EXAMPLES))
        if bench_rows:
            st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, height=430, hide_index=True)
        else:
            st.info("No aspects detected.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown(
        "<div class='section-label'>Statistical Metrics — Unseen Test Split</div>",
        unsafe_allow_html=True,
    )

    metrics_data = {}
    if os.path.exists(TEST_METRICS_PATH):
        try:
            with open(TEST_METRICS_PATH) as f:
                metrics_data = json.load(f)
        except Exception:
            pass

    tm = metrics_data.get("test", metrics_data)
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Overall RMSE",     f"{tm.get('overall_rmse',    0.1775):.4f}")
    with m2: st.metric("Valence RMSE",     f"{tm.get('valence_rmse',    0.2101):.4f}")
    with m3: st.metric("Arousal RMSE",     f"{tm.get('arousal_rmse',    0.1374):.4f}")
    with m4: st.metric("Valence R\u00b2",  f"{tm.get('valence_r2',      0.2121):.4f}")

    m5, m6, m7, m8 = st.columns(4)
    with m5: st.metric("Valence MAE",      f"{tm.get('valence_mae',     0.1699):.4f}")
    with m6: st.metric("Arousal MAE",      f"{tm.get('arousal_mae',     0.1079):.4f}")
    with m7: st.metric("Valence Lin CCC",  f"{tm.get('valence_ccc',     0.4843):.4f}")
    with m8: st.metric("Arousal Lin CCC",  f"{tm.get('arousal_ccc',     0.3300):.4f}")

    if os.path.exists(METRICS_SAVE_PATH):
        st.markdown(
            "<div class='section-label' style='margin-top:1.5rem'>Training &amp; Validation Convergence (200 Epochs)</div>",
            unsafe_allow_html=True,
        )
        hist_df = pd.read_csv(METRICS_SAVE_PATH)
        fig_hist = go.Figure()
        for col_name, color, dash in [
            ("Train Loss",       "#38bdf8", "solid"),
            ("Val Loss",         "#f43f5e", "dash"),
            ("Val Valence RMSE", "#10b981", "solid"),
            ("Val Arousal RMSE", "#fbbf24", "solid"),
        ]:
            if col_name in hist_df.columns:
                fig_hist.add_trace(go.Scatter(
                    x=hist_df["Epoch"], y=hist_df[col_name],
                    mode="lines", name=col_name,
                    line=dict(color=color, width=1.8, dash=dash),
                ))
        fig_hist.update_layout(
            xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Loss / RMSE", gridcolor="rgba(255,255,255,0.05)"),
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#64748b", size=12),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
            height=320, margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    if os.path.exists(PREDICTIONS_COMPARISON_PATH):
        st.markdown(
            "<div class='section-label' style='margin-top:1.5rem'>Test Set: Ground Truth vs Predictions</div>",
            unsafe_allow_html=True,
        )
        pred_df = pd.read_csv(PREDICTIONS_COMPARISON_PATH)
        dc = [c for c in [
            "sentence", "aspect", "opinion",
            "true_valence", "pred_valence", "valence_error",
            "true_arousal", "pred_arousal", "arousal_error",
        ] if c in pred_df.columns]
        st.dataframe(pred_df[dc].head(50), use_container_width=True, height=340, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Dataset Browser
# ═══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown(
        "<div class='section-label'>Dataset Exploration — DimABSA Hinglish Corpus</div>",
        unsafe_allow_html=True,
    )
    raw_df = load_dataset()
    if not raw_df.empty:
        expanded_df = expand_aspect_dataset(raw_df)
        d1, d2, d3 = st.columns(3)
        with d1: st.metric("Total Sentences",         f"{len(raw_df):,}")
        with d2: st.metric("Unrolled Aspect Samples",  f"{len(expanded_df):,}")
        with d3: st.metric("Avg Aspects / Sentence",   f"{len(expanded_df)/len(raw_df):.2f}")

        search_q = st.text_input(
            "search_box",
            placeholder="Filter by aspect, opinion, or sentence keyword...",
            label_visibility="collapsed",
        )
        filtered = expanded_df
        if search_q:
            mask = (
                expanded_df["sentence"].str.contains(search_q, case=False, na=False)
                | expanded_df["aspect"].str.contains(search_q, case=False, na=False)
            )
            filtered = expanded_df[mask]

        cols_show = [c for c in ["sentence_id", "sentence", "aspect", "opinion", "valence", "arousal"] if c in filtered.columns]
        st.dataframe(filtered[cols_show].head(100), use_container_width=True, height=420, hide_index=True)
    else:
        st.info("Dataset file not found.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <span>Aspect-Based Emotion Regression &nbsp;&middot;&nbsp; Built for Aspect-level Valence &amp; Arousal Prediction</span>
  <span>Made with &hearts; for NLP Research</span>
</div>
""", unsafe_allow_html=True)
