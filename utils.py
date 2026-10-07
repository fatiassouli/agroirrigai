# =============================================================================
# utils.py — Visualisations Plotly, CSS beige high-class, historique, export
# =============================================================================

import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import CLASS_COLORS, CLASS_ICONS, HISTORY_DISPLAY_COLS, MAX_HISTORY, THEME

logger = logging.getLogger(__name__)

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=THEME["text"], family="'Lato', Georgia, serif"),
    margin=dict(l=20, r=20, t=44, b=20),
)

# =============================================================================
# VISUALISATIONS PRÉDICTION
# =============================================================================

def build_risk_gauge(label: str, confidence: float) -> go.Figure:
    val_map = {"Low": 18, "Medium": 52, "High": 86}
    val   = val_map.get(label, 50)
    color = CLASS_COLORS[label]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number=dict(suffix=" %", font=dict(size=30, color=color)),
        title=dict(
            text=(
                f"{CLASS_ICONS.get(label,'')}  {label}"
                f"<br><span style='font-size:0.8em;color:{THEME['muted']}'>"
                f"Confiance : {confidence:.1%}</span>"
            ),
            font=dict(size=15, color=THEME["text"]),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickvals=[0, 33, 66, 100],
                ticktext=["0", "Low", "Medium", "High"],
                tickcolor=THEME["muted"],
                tickwidth=1,
            ),
            bar=dict(color=color, thickness=0.28),
            bgcolor=THEME["surface"],
            borderwidth=0,
            steps=[
                dict(range=[0,  33], color="#dff0d0"),
                dict(range=[33, 66], color="#f5e6cc"),
                dict(range=[66,100], color="#f5d8d0"),
            ],
            threshold=dict(
                line=dict(color=THEME["text"], width=3),
                thickness=0.75,
                value=val,
            ),
        ),
    ))
    fig.update_layout(**PLOTLY_BASE, height=270)
    return fig


def build_probability_bar(probabilities: Dict[str, float]) -> go.Figure:
    labels = list(probabilities.keys())
    values = [v * 100 for v in probabilities.values()]
    colors = [CLASS_COLORS[l] for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color=THEME["text"], size=12, family="Lato"),
        hovertemplate="<b>%{y}</b> : %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text="Probabilités par classe", font=dict(size=13, color=THEME["text"])),
        xaxis=dict(
            range=[0, 118], ticksuffix="%",
            showgrid=True, gridcolor=THEME["border"],
            color=THEME["text"],
        ),
        yaxis=dict(showgrid=False, color=THEME["text"]),
        height=210,
    )
    return fig


def build_probability_donut(probabilities: Dict[str, float]) -> go.Figure:
    labels = list(probabilities.keys())
    fig = go.Figure(go.Pie(
        labels=labels,
        values=list(probabilities.values()),
        hole=0.56,
        marker=dict(
            colors=[CLASS_COLORS[l] for l in labels],
            line=dict(color=THEME["background"], width=3),
        ),
        textinfo="label+percent",
        textfont=dict(color=THEME["text"], size=12),
        hovertemplate="<b>%{label}</b> : %{percent}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text="Répartition des probabilités", font=dict(size=13, color=THEME["text"])),
        legend=dict(orientation="h", y=-0.15, font=dict(color=THEME["text"])),
        height=250,
    )
    return fig


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def build_feature_importance_chart(features: List[tuple]) -> go.Figure:
    names = [f[0] for f in features]
    vals  = [f[1] for f in features]
    n     = len(names)

    # Palette beige → vert doux → vert soutenu
    colors = px.colors.sample_colorscale(
        [[0, "#a3b18a"], [0.5, "#6a994e"], [1, "#3a6b2a"]],
        [i / max(n - 1, 1) for i in range(n)],
    )[::-1]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
        textfont=dict(color=THEME["text"], size=10),
        hovertemplate="<b>%{y}</b><br>Importance : %{x:.6f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text="🏆 Feature Importances — XGBoost", font=dict(size=15, color=THEME["text"])),
        xaxis=dict(
            title="Importance (gain)",
            showgrid=True, gridcolor=THEME["border"],
            color=THEME["text"],
        ),
        yaxis=dict(autorange="reversed", showgrid=False, color=THEME["text"]),
        height=max(320, n * 28 + 80),
    )
    return fig


def build_confusion_matrix_chart(cm: List[List[int]], labels: List[str]) -> go.Figure:
    import numpy as np
    cm_arr  = np.array(cm)
    cm_norm = cm_arr / cm_arr.sum(axis=1, keepdims=True)

    text = [
        [f"{cm_arr[i][j]}<br>({cm_norm[i][j]:.1%})" for j in range(len(labels))]
        for i in range(len(labels))
    ]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=[f"Prédit : {l}" for l in labels],
        y=[f"Réel : {l}"   for l in labels],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=12, color=THEME["text"]),
        colorscale=[[0, "#f5f1e6"], [0.5, "#a3b18a"], [1, "#3a6b2a"]],
        showscale=True,
        hoverongaps=False,
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text="Matrice de Confusion — XGBoost", font=dict(size=14, color=THEME["text"])),
        xaxis=dict(side="bottom", color=THEME["text"]),
        yaxis=dict(color=THEME["text"]),
        height=340,
    )
    return fig


# =============================================================================
# HISTORIQUE
# =============================================================================

def init_history() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def add_to_history(raw_input: Dict[str, Any], result: Dict[str, Any]) -> None:
    init_history()
    record = {
        "Timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **raw_input,
        "Prediction": result["prediction_label"],
        "P(Low)":     round(result["probabilities"]["Low"],    3),
        "P(Medium)":  round(result["probabilities"]["Medium"],  3),
        "P(High)":    round(result["probabilities"]["High"],    3),
        "Confidence": round(result["confidence"],               3),
    }
    st.session_state.prediction_history.insert(0, record)
    if len(st.session_state.prediction_history) > MAX_HISTORY:
        st.session_state.prediction_history = st.session_state.prediction_history[:MAX_HISTORY]


def get_history_df() -> Optional[pd.DataFrame]:
    init_history()
    if not st.session_state.prediction_history:
        return None
    return pd.DataFrame(st.session_state.prediction_history)


def build_history_trend(df: pd.DataFrame) -> go.Figure:
    d = df[["Timestamp", "P(Low)", "P(Medium)", "P(High)"]].iloc[::-1].reset_index(drop=True)
    fig = go.Figure()
    for cls, color in CLASS_COLORS.items():
        fig.add_trace(go.Scatter(
            x=d["Timestamp"], y=d[f"P({cls})"],
            name=cls, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5, line=dict(color=THEME["background"], width=1)),
        ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text="Évolution des probabilités", font=dict(size=13, color=THEME["text"])),
        xaxis=dict(
            showgrid=True, gridcolor=THEME["border"],
            tickangle=30, color=THEME["text"],
        ),
        yaxis=dict(
            range=[0, 1],
            showgrid=True, gridcolor=THEME["border"],
            color=THEME["text"],
        ),
        legend=dict(orientation="h", y=1.12, font=dict(color=THEME["text"])),
        height=280,
    )
    return fig


def build_history_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["Prediction"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.42,
        marker=dict(
            colors=[CLASS_COLORS.get(l, "#888") for l in counts.index],
            line=dict(color=THEME["background"], width=3),
        ),
        textinfo="label+value+percent",
        textfont=dict(color=THEME["text"]),
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text="Distribution des prédictions", font=dict(size=13, color=THEME["text"])),
        legend=dict(font=dict(color=THEME["text"])),
        height=270,
    )
    return fig


# =============================================================================
# EXPORT
# =============================================================================

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")


# =============================================================================
# CSS PALETTE BEIGE HIGH-CLASS
# =============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;600;700&family=Playfair+Display:ital,wght@0,600;0,700;0,800;1,600&display=swap');

/* ── Reset global ────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Lato', sans-serif !important;
    color: #2f2a23 !important;
}

/* ── Fond global beige ───────────────────────────────────────────── */
.stApp {
    background: linear-gradient(160deg, #f5f1e6 0%, #efe9d8 60%, #f5f1e6 100%) !important;
}

/* ── Sidebar beige sablé ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #ede5d0 !important;
    border-right: 1px solid #d6c7a8 !important;
}
[data-testid="stSidebar"] * {
    color: #2f2a23 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid #c8b99a !important;
    color: #2f2a23 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all .2s;
    text-align: left !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #e0d5be !important;
    border-color: #6a994e !important;
    color: #6a994e !important;
}

/* ── Titre gradient nature ───────────────────────────────────────── */
.main-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4a7c2f, #c77b2f, #5c7a9e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: .3rem;
    letter-spacing: -.01em;
}

/* ── Section headers ─────────────────────────────────────────────── */
.sec-hdr {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.08rem;
    font-weight: 700;
    color: #2f2a23 !important;
    border-bottom: 2px solid #6a994e;
    padding-bottom: .4rem;
    margin-bottom: 1rem;
    letter-spacing: .01em;
}

/* ── About section headers ───────────────────────────────────────── */
.about-hdr {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.3rem;
    font-weight: 700;
    color: #9b4d2a !important;
    margin: 1.5rem 0 .6rem;
    border-left: 4px solid #c77b2f;
    padding-left: .7rem;
}

/* ── KPI / nat-card ──────────────────────────────────────────────── */
.nat-card {
    background: #efe7d6;
    border: 1px solid #d6c7a8;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
    box-shadow: 0 2px 8px rgba(100,80,40,.08);
}
.nat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(100,80,40,.16);
}
.nat-card .val {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.2;
}
.nat-card .lbl {
    font-size: 0.72rem;
    color: #6b6256 !important;
    margin-top: .3rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 600;
}

/* ── Metric badges (About page) ──────────────────────────────────── */
.metric-badge {
    background: #efe7d6;
    border: 1px solid #d6c7a8;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(100,80,40,.08);
}
.metric-badge .mv {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 800;
}
.metric-badge .ml {
    font-size: .72rem;
    color: #6b6256 !important;
    text-transform: uppercase;
    letter-spacing: .07em;
    font-weight: 600;
}

/* ── Pred result box ─────────────────────────────────────────────── */
.pred-box {
    border-radius: 16px;
    padding: 1.4rem 2rem;
    text-align: center;
    border: 2px solid;
    animation: fadeUp .4s ease;
    box-shadow: 0 4px 20px rgba(100,80,40,.12);
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(12px); }
    to   { opacity:1; transform:translateY(0); }
}
.pred-Low    { border-color:#6a994e; background: rgba(106,153,78,.10); }
.pred-Medium { border-color:#c77b2f; background: rgba(199,123,47,.10); }
.pred-High   { border-color:#9b2226; background: rgba(155,34,38,.10); }

/* ── Pipeline boxes ──────────────────────────────────────────────── */
.pipe-step {
    background: #efe7d6;
    border: 1px solid #d6c7a8;
    border-radius: 10px;
    padding: .7rem 1rem;
    text-align: center;
    font-size: .85rem;
    color: #2f2a23;
    font-weight: 700;
    flex: 1;
}
.pipe-arrow { color: #6a994e; font-size: 1.4rem; align-self: center; }

/* ── Bouton principal ────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #4a7c59, #6a994e) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: .7rem 2rem !important;
    color: #f5f1e6 !important;
    letter-spacing: .03em;
    transition: opacity .2s, transform .15s;
    box-shadow: 0 4px 14px rgba(74,124,89,.3) !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: .9;
    transform: translateY(-1px);
}

/* ── Inputs ──────────────────────────────────────────────────────── */
/* Champs numériques et texte */

.stNumberInput input,
.stTextInput input {
    background-color: #F5E9C4 !important;
    border: 1px solid #c8b99a !important;
    color: #2f2a23 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Liste déroulante */

[data-baseweb="select"] > div {
    background-color: #F5E9C4 !important;
    border: 1px solid #c8b99a !important;
    border-radius: 8px !important;
    color: #2f2a23 !important;
    font-weight: 600 !important;
}

/* Focus */

.stNumberInput input:focus,
.stTextInput input:focus,
[data-baseweb="select"] > div:focus-within {
    border-color: #6a994e !important;
    box-shadow: 0 0 0 2px rgba(106,153,78,.18) !important;
}

/* ── Tabs ────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background: #e8ddc7 !important;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
[data-baseweb="tab"] {
    border-radius: 9px !important;
    color: #6b6256 !important;
    font-weight: 600 !important;
}
[aria-selected="true"] {
    background: #6a994e !important;
    color: #f5f1e6 !important;
}

/* ── Dataframe / Table ───────────────────────────────────────────── */
.stDataFrame {
    border: 1px solid #d6c7a8 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Info / warning / success / error boxes ──────────────────────── */
.stAlert {
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* ── Expander ────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #e8ddc7 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    color: #2f2a23 !important;
}

/* ── Scrollbar nature ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #efe7d6; }
::-webkit-scrollbar-thumb { background: #c8b99a; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #6a994e; }

/* ── HR ──────────────────────────────────────────────────────────── */
hr { border-color: #d6c7a8 !important; }

/* ── Metric Streamlit natif ──────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #efe7d6;
    border: 1px solid #d6c7a8;
    border-radius: 10px;
    padding: .7rem 1rem;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-weight: 800 !important;
    color: #2f2a23 !important;
}
[data-testid="stMetricLabel"] {
    font-weight: 700 !important;
    color: #6b6256 !important;
    text-transform: uppercase;
    font-size: .72rem;
    letter-spacing: .07em;
}

/* ── Caption ─────────────────────────────────────────────────────── */
.stCaption, .css-1fcdlhc { color: #6b6256 !important; }

/* ── Slider ──────────────────────────────────────────────────────── */
[data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #2f2a23 !important;
    font-weight: 700 !important;
}

/* ── Radio ───────────────────────────────────────────────────────── */
.stRadio label { font-weight: 600 !important; color: #2f2a23 !important; }
</style>
"""


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# COMPOSANTS HTML
# =============================================================================

def kpi_card(value: str, label: str, color: str) -> str:
    return f"""<div class="nat-card">
        <div class="val" style="color:{color};">{value}</div>
        <div class="lbl">{label}</div>
    </div>"""


def pred_card(label: str, confidence: float) -> str:
    color = CLASS_COLORS.get(label, "#888")
    icon  = CLASS_ICONS.get(label, "🌱")
    return f"""<div class="pred-box pred-{label}">
        <div style="font-size:2.8rem;line-height:1;">{icon}</div>
        <div style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:800;
                    color:{color};margin-top:.4rem;letter-spacing:-.01em;">{label}</div>
        <div style="color:#6b6256;margin-top:.4rem;font-weight:600;font-size:.92rem;">
            Confiance : <strong style="color:{color};">{confidence:.1%}</strong>
        </div>
    </div>"""


def section_header(text: str) -> None:
    st.markdown(f'<div class="sec-hdr">{text}</div>', unsafe_allow_html=True)


def about_header(text: str) -> None:
    st.markdown(f'<div class="about-hdr">{text}</div>', unsafe_allow_html=True)
