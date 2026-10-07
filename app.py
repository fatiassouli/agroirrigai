# =============================================================================
# app.py —  AquaFarm — Application Streamlit
# Pages : Prediction Manuelle | Import CSV | Historique | Prediction Meteo | A propos
# =============================================================================

import logging
import sys
from datetime import datetime
from typing import Any, Dict, List
from weather_service import get_weather
from cities import CITIES
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import THEME


st.set_page_config(
    page_title="AquaFarm",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


from config import (
    CLASS_COLORS,
    CLASS_ICONS,
    CROP_GROWTH_STAGE_MAP,
    CROP_GROWTH_STAGE_LABELS,
    CROP_TYPE_LABELS,
    CROP_TYPES,
    HISTORY_DISPLAY_COLS,
    IRRIGATION_TYPE_LABELS,
    IRRIGATION_TYPES,
    MULCHING_LABELS,
    NUMERIC_FEATURES,
    REGION_LABELS,
    REGIONS,
    SEASON_LABELS,
    SEASONS,
    SOIL_TYPE_LABELS,
    SOIL_TYPES,
    THEME,
    WATER_SOURCE_LABELS,
    WATER_SOURCES,
)

from prediction import get_feature_importance, load_model, predict, predict_batch
from utils import (
    about_header, add_to_history, build_confusion_matrix_chart,
    build_feature_importance_chart, build_history_distribution,
    build_history_trend, build_probability_bar, build_probability_donut,
    build_risk_gauge, df_to_csv_bytes, get_history_df,
    inject_css, init_history, kpi_card, pred_card, section_header,
)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

PLOTLY_CFG = {"displayModeBar": False}


# =============================================================================
# INITIALISATION SESSION
# =============================================================================

def init_session():
    init_history()
    for k, v in [("prediction_count", 0), ("last_result", None), ("current_page", "manual")]:
        if k not in st.session_state:
            st.session_state[k] = v


# =============================================================================
# BARRE LATERALE
# =============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:1.2rem 0 .8rem;">
            <div style="font-size:2.8rem;">🌿</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.45rem;font-weight:800;
                        background:linear-gradient(90deg,#4a7c2f,#c77b2f,#5c7a9e);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;letter-spacing:-.01em;">AquaFarm</div>
            <div style="font-size:.72rem;color:{THEME['muted']};margin-top:.3rem;
                        font-weight:600;text-transform:uppercase;letter-spacing:.08em;">
                Aide à la décision irrigation
            </div>
        </div>
        <hr style="border-color:{THEME['border']};margin:.5rem 0;">
        """, unsafe_allow_html=True)

        pages = {
            " Prediction Manuelle":          "manual",
            " Prediction Meteo":             "smart",
            " Importer un fichier CSV":      "csv",
            " Historique":                   "history",
        }

        st.markdown(
            f"<div style='font-size:.68rem;color:{THEME['primary']};font-weight:700;"
            f"text-transform:uppercase;letter-spacing:.12em;margin-bottom:.5rem;'>"
            f"Navigation</div>",
            unsafe_allow_html=True,
        )

        for label, pid in pages.items():
            if st.button(label, key=f"nav_{pid}", use_container_width=True):
                st.session_state.current_page = pid
                st.rerun()

        # Statistiques de session
        st.markdown(f"<hr style='border-color:{THEME['border']};'>", unsafe_allow_html=True)
        df_h = get_history_df()
        n = len(df_h) if df_h is not None else 0
        col_a, col_b = st.columns(2)
        col_a.metric("Prédictions", n)
        if df_h is not None and n > 0:
            last = df_h.iloc[0]["Prediction"]
            col_b.metric("Dernière", CLASS_ICONS.get(last, "") + " " + last)
        else:
            col_b.metric("Dernière", "—")

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("© 2026 AquaFarm — Master IA")


# =============================================================================
# PAGE 1 : PREDICTION MANUELLE
# =============================================================================

def page_manual(model, feature_names):
    st.markdown('<div class="main-title"> Prediction Manuelle</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:{THEME["muted"]};font-weight:500;margin-bottom:1.4rem;">'
        f'Renseignez les paramètres agronomiques pour prédire le besoin en irrigation.</div>',
        unsafe_allow_html=True,
    )

    def num(col, key):
        cfg = NUMERIC_FEATURES[key]
        lbl = f"{cfg.label} ({cfg.unit})" if cfg.unit else cfg.label
        return col.number_input(lbl, min_value=float(cfg.min_val), max_value=float(cfg.max_val),
                                value=float(cfg.default), step=float(cfg.step),
                                help=cfg.help_text, key=f"inp_{key}")

    with st.form("form_predict", clear_on_submit=False):

        # Parametres sol
        section_header(" Parametres Sol")
        c1, c2, c3, c4 = st.columns(4)
        soil_ph     = num(c1, "Soil_pH")
        soil_moist  = num(c2, "Soil_Moisture")
        org_carbon  = num(c3, "Organic_Carbon")
        elec_cond   = num(c4, "Electrical_Conductivity")

        c1, c2 = st.columns(2)
        soil_type = c1.selectbox(
            " Type de sol", SOIL_TYPES,
            format_func=lambda x: SOIL_TYPE_LABELS[x],
            key="s_soil"
        )
        mulching = c2.radio(
            " Paillage utilisé ?",
            options=["No", "Yes"],
            format_func=lambda x: MULCHING_LABELS[x],
            horizontal=True, key="s_mulch"
        )

        # Parametres climatiques
        st.markdown("<br>", unsafe_allow_html=True)
        section_header(" Parametres Climatiques")
        c1, c2, c3, c4 = st.columns(4)
        temperature = num(c1, "Temperature_C")
        humidity    = num(c2, "Humidity")
        rainfall    = num(c3, "Rainfall_mm")
        sunlight    = num(c4, "Sunlight_Hours")
        c1, c2, _, _ = st.columns(4)
        wind_speed  = num(c1, "Wind_Speed_kmh")

        # Parametres culture
        st.markdown("<br>", unsafe_allow_html=True)
        section_header(" Culture & Parcelle")
        c1, c2, c3, c4 = st.columns(4)
        crop_type = c1.selectbox(
            " Type de culture", CROP_TYPES,
            format_func=lambda x: CROP_TYPE_LABELS[x],
            key="s_crop"
        )
        crop_stage = c2.selectbox(
            " Stade de croissance",
            options=list(CROP_GROWTH_STAGE_MAP.keys()),
            format_func=lambda x: CROP_GROWTH_STAGE_LABELS[x],
            key="s_stage"
        )
        field_area  = num(c3, "Field_Area_hectare")
        prev_irrig  = num(c4, "Previous_Irrigation_mm")

        # Contexte environnemental
        st.markdown("<br>", unsafe_allow_html=True)
        section_header(" Contexte Environnemental")
        c1, c2, c3, c4 = st.columns(4)

        season = c1.selectbox(
            " Saison", SEASONS,
            format_func=lambda x: SEASON_LABELS[x],
            key="s_season"
        )
        irrig_type = c2.selectbox(
            " Type d'irrigation", IRRIGATION_TYPES,
            format_func=lambda x: IRRIGATION_TYPE_LABELS[x],
            key="s_itype"
        )
        water_source = c3.selectbox(
            " Source d'eau", WATER_SOURCES,
            format_func=lambda x: WATER_SOURCE_LABELS[x],
            key="s_wsource"
        )
        region = c4.selectbox(
            " Région", REGIONS,
            format_func=lambda x: REGION_LABELS[x],
            key="s_region"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            " Lancer la Prediction",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        raw = {
            "Soil_pH":                   soil_ph,
            "Soil_Moisture":             soil_moist,
            "Organic_Carbon":            org_carbon,
            "Electrical_Conductivity":   elec_cond,
            "Temperature_C":             temperature,
            "Humidity":                  humidity,
            "Rainfall_mm":               rainfall,
            "Sunlight_Hours":            sunlight,
            "Wind_Speed_kmh":            wind_speed,
            "Crop_Growth_Stage":         crop_stage,
            "Field_Area_hectare":        field_area,
            "Mulching_Used":             mulching,
            "Previous_Irrigation_mm":    prev_irrig,
            "Soil_Type":                 soil_type,
            "Crop_Type":                 crop_type,
            "Season":                    season,
            "Irrigation_Type":           irrig_type,
            "Water_Source":              water_source,
            "Region":                    region,
        }

        with st.spinner(" Analyse en cours…"):
            try:
                result = predict(model, feature_names, raw)
            except Exception as exc:
                st.error(f" Erreur : {exc}")
                logger.exception("Erreur prédiction")
                return

        add_to_history(raw, result)
        st.session_state.prediction_count += 1
        _show_result(result)


def _show_result(result):
    label      = result["prediction_label"]
    confidence = result["confidence"]
    probs      = result["probabilities"]

    st.markdown("<br>", unsafe_allow_html=True)
    section_header(" Résultats de la Prediction")

    c_card, c_gauge, c_bars = st.columns([1.5, 2, 2])

    with c_card:
        st.markdown(pred_card(label, confidence), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        advice = {
            "Low":    ("Aucune irrigation requise. Le sol dispose de suffisamment d'humidité.", "success"),
            "Medium": ("Irrigation légère conseillée dans les 24–48 h.", "warning"),
            "High":   ("Irrigation urgente ! La culture est en stress hydrique sévère.", "error"),
        }
        msg, fn = advice[label]
        getattr(st, fn)(msg)

    with c_gauge:
        st.plotly_chart(build_risk_gauge(label, confidence),
                        use_container_width=True, config=PLOTLY_CFG)

    with c_bars:
        st.plotly_chart(build_probability_bar(probs),
                        use_container_width=True, config=PLOTLY_CFG)
        st.plotly_chart(build_probability_donut(probs),
                        use_container_width=True, config=PLOTLY_CFG)

    df_p = pd.DataFrame([
        {"Classe":      f"{CLASS_ICONS.get(k, '')} {k}",
         "Probabilité": f"{v:.4f}",
         "Pourcentage": f"{v:.1%}",
         "Statut":      "Prédite" if k == label else ""}
        for k, v in probs.items()
    ])
    st.dataframe(df_p, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 2 : PREDICTION METEO (donnees climatiques depuis OpenWeather)
# =============================================================================

def page_smart(model, feature_names):
    st.markdown('<div class="main-title"> Prediction Meteo</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:{THEME["muted"]};font-weight:500;margin-bottom:1.4rem;">'
        f'Les paramètres climatiques sont récupérés automatiquement via OpenWeather. '
        f'Renseignez uniquement les paramètres agronomiques de la parcelle.</div>',
        unsafe_allow_html=True,
    )

    # --- Saisie de la ville et recuperation meteo ---
# --- Sélection de la ville et récupération météo ---
    section_header(" Localisation")

    st.markdown(
        """
        <style>
        div.st-key-weather_fetch_btn {
            margin-top: 1.75rem;
        }
        div.st-key-weather_fetch_btn button[kind="primary"] {
            background-color: rgb(106, 153, 78) !important;
            border-color: rgb(106, 153, 78) !important;
        }
        div.st-key-weather_fetch_btn button[kind="primary"]:hover {
            background-color: rgb(92, 134, 66) !important;
            border-color: rgb(92, 134, 66) !important;
        }
        div.st-key-weather_fetch_btn button[kind="primary"]:active {
            background-color: rgb(80, 118, 57) !important;
            border-color: rgb(80, 118, 57) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_city, col_btn = st.columns([3, 1])

    default_index = (
        CITIES.index(st.session_state.get("smart_city"))
        if st.session_state.get("smart_city") in CITIES
        else 0
)

    city = col_city.selectbox(
        "Ville",
        options=CITIES,
        index=default_index,
        key="smart_city_select",
)

    with col_btn:
        with st.container(key="weather_fetch_btn"):
            fetch_clicked = st.button(
                "Récupérer la météo",
                type="primary",
                use_container_width=True,
            )

    if fetch_clicked:
        with st.spinner("Récupération des données météo…"):
            try:
                weather = get_weather(city)
                st.session_state["smart_weather"] = weather
                st.session_state["smart_city"] = city

            except Exception as exc:
                st.error(f"Impossible de récupérer la météo : {exc}")
                logger.exception("Erreur weather API")
                st.stop()

    weather = st.session_state.get("smart_weather")

    if weather:
        st.success(
            f"Données météo chargées pour **{st.session_state.get('smart_city', '')}**"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        section_header(" Parametres Climatiques (depuis OpenWeather)")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Temperature (°C)",   weather["Temperature_C"])
        c2.metric("Humidité (%)",        weather["Humidity"])
        c3.metric("Précipitations (mm)", weather["Rainfall_mm"])
        c4.metric("Vent (km/h)",         f"{weather['Wind_Speed_kmh']:.1f}")
        c5.metric("Ensoleillement (h)",  weather["Sunlight_Hours"])
    else:
        st.info("Saisissez une ville puis cliquez sur 'Récupérer la météo' pour charger les données climatiques.")

    # --- Formulaire des parametres agronomiques ---
    st.markdown("<br>", unsafe_allow_html=True)

    def num(col, key, suffix="sm"):
        cfg = NUMERIC_FEATURES[key]
        lbl = f"{cfg.label} ({cfg.unit})" if cfg.unit else cfg.label
        return col.number_input(
            lbl,
            min_value=float(cfg.min_val), max_value=float(cfg.max_val),
            value=float(cfg.default), step=float(cfg.step),
            help=cfg.help_text, key=f"sm_{key}",
        )

    with st.form("form_smart", clear_on_submit=False):

        section_header(" Parametres Sol")
        c1, c2, c3, c4 = st.columns(4)
        soil_ph    = num(c1, "Soil_pH")
        soil_moist = num(c2, "Soil_Moisture")
        org_carbon = num(c3, "Organic_Carbon")
        elec_cond  = num(c4, "Electrical_Conductivity")

        c1, c2 = st.columns(2)
        soil_type = c1.selectbox(
            "Type de sol", SOIL_TYPES,
            format_func=lambda x: SOIL_TYPE_LABELS[x],
            key="sm_soil",
        )
        mulching = c2.radio(
            "Paillage utilisé ?",
            options=["No", "Yes"],
            format_func=lambda x: MULCHING_LABELS[x],
            horizontal=True, key="sm_mulch",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        section_header(" Culture & Parcelle")
        c1, c2, c3, c4 = st.columns(4)
        crop_type = c1.selectbox(
            "Type de culture", CROP_TYPES,
            format_func=lambda x: CROP_TYPE_LABELS[x],
            key="sm_crop",
        )
        crop_stage = c2.selectbox(
            "Stade de croissance",
            options=list(CROP_GROWTH_STAGE_MAP.keys()),
            format_func=lambda x: CROP_GROWTH_STAGE_LABELS[x],
            key="sm_stage",
        )
        field_area = num(c3, "Field_Area_hectare")
        prev_irrig = num(c4, "Previous_Irrigation_mm")

        st.markdown("<br>", unsafe_allow_html=True)
        section_header(" Contexte Environnemental")
        c1, c2, c3, c4 = st.columns(4)
        season = c1.selectbox(
            "Saison", SEASONS,
            format_func=lambda x: SEASON_LABELS[x],
            key="sm_season",
        )
        irrig_type = c2.selectbox(
            "Type d'irrigation", IRRIGATION_TYPES,
            format_func=lambda x: IRRIGATION_TYPE_LABELS[x],
            key="sm_itype",
        )
        water_source = c3.selectbox(
            "Source d'eau", WATER_SOURCES,
            format_func=lambda x: WATER_SOURCE_LABELS[x],
            key="sm_wsource",
        )
        region = c4.selectbox(
            "Région", REGIONS,
            format_func=lambda x: REGION_LABELS[x],
            key="sm_region",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            " Lancer la Prediction",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not weather:
            st.error("Veuillez d'abord récupérer les données météo en saisissant une ville.")
            return

        raw = {
            "Soil_pH":                   soil_ph,
            "Soil_Moisture":             soil_moist,
            "Organic_Carbon":            org_carbon,
            "Electrical_Conductivity":   elec_cond,
            "Temperature_C":             weather["Temperature_C"],
            "Humidity":                  weather["Humidity"],
            "Rainfall_mm":               weather["Rainfall_mm"],
            "Sunlight_Hours":            weather["Sunlight_Hours"],
            "Wind_Speed_kmh":            weather["Wind_Speed_kmh"],
            "Crop_Growth_Stage":         crop_stage,
            "Field_Area_hectare":        field_area,
            "Mulching_Used":             mulching,
            "Previous_Irrigation_mm":    prev_irrig,
            "Soil_Type":                 soil_type,
            "Crop_Type":                 crop_type,
            "Season":                    season,
            "Irrigation_Type":           irrig_type,
            "Water_Source":              water_source,
            "Region":                    region,
        }

        with st.spinner(" Analyse en cours…"):
            try:
                result = predict(model, feature_names, raw)
            except Exception as exc:
                st.error(f" Erreur : {exc}")
                logger.exception("Erreur prédiction smart")
                return

        add_to_history(raw, result)
        st.session_state.prediction_count += 1
        _show_result(result)


# =============================================================================
# PAGE 3 : IMPORT CSV
# =============================================================================

def page_csv(model, feature_names):
    st.markdown('<div class="main-title"> Prediction par lot</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:{THEME["muted"]};font-weight:500;margin-bottom:1.4rem;">'
        f'Importez un fichier CSV pour obtenir des prédictions en masse.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(" Fichier CSV", type=["csv"])

    if uploaded is None:
        st.info(
            "Le fichier doit contenir les colonnes brutes du jeu de données "
            "(avant encodage). Colonnes numériques et catégorielles originales."
        )
        cols_req = list(NUMERIC_FEATURES.keys()) + [
            "Crop_Growth_Stage", "Mulching_Used",
            "Soil_Type", "Crop_Type", "Season", "Irrigation_Type", "Water_Source", "Region"
        ]
        st.dataframe(pd.DataFrame({"Colonne requise": cols_req}),
                     use_container_width=True, hide_index=True)
        return

    try:
        df_up = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Lecture impossible : {exc}")
        return

    st.success(f"{len(df_up)} lignes · {len(df_up.columns)} colonnes chargées.")

    with st.expander(" Aperçu", expanded=False):
        st.dataframe(df_up.head(8), use_container_width=True)

    if st.button(" Lancer les prédictions", type="primary"):
        with st.spinner(f"Prédiction sur {len(df_up)} lignes…"):
            try:
                df_res = predict_batch(model, feature_names, df_up)
            except Exception as exc:
                st.error(f"{exc}")
                return

        st.success("Prédictions terminées !")

        n_low = (df_res["Prediction"] == "Low").sum()
        n_med = (df_res["Prediction"] == "Medium").sum()
        n_hi  = (df_res["Prediction"] == "High").sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi_card(str(len(df_res)), "Total",          "#5c7a9e"),           unsafe_allow_html=True)
        c2.markdown(kpi_card(str(n_low),  " Faible",           CLASS_COLORS["Low"]),  unsafe_allow_html=True)
        c3.markdown(kpi_card(str(n_med),  " Modéré",           CLASS_COLORS["Medium"]), unsafe_allow_html=True)
        c4.markdown(kpi_card(str(n_hi),   " Élevé",            CLASS_COLORS["High"]),  unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(build_history_distribution(df_res),
                            use_container_width=True, config=PLOTLY_CFG)
        with ch2:
            counts = df_res["Prediction"].value_counts()
            fig = go.Figure(go.Bar(
                x=counts.index.tolist(), y=counts.values.tolist(),
                marker_color=[CLASS_COLORS.get(l, "#888") for l in counts.index],
                text=counts.values.tolist(), textposition="outside",
                textfont=dict(color=THEME["text"], size=12),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=THEME["text"]),
                title=dict(text="Distribution du lot", font=dict(size=13, color=THEME["text"])),
                height=260,
                xaxis=dict(showgrid=False, color=THEME["text"]),
                yaxis=dict(showgrid=True, gridcolor=THEME["border"], color=THEME["text"]),
                margin=dict(l=20, r=20, t=44, b=20),
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

        section_header(" Résultats")
        priority = ["Prediction", "P(Low)", "P(Medium)", "P(High)", "Confidence"]
        rest = [c for c in df_res.columns if c not in priority]
        st.dataframe(df_res[priority + rest].head(200), use_container_width=True)

        st.download_button(
            "Télécharger les résultats (CSV)",
            data=df_to_csv_bytes(df_res),
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", type="primary",
        )


# =============================================================================
# PAGE 4 : HISTORIQUE
# =============================================================================

def page_history():
    st.markdown('<div class="main-title"> Historique des Predictions</div>',
                unsafe_allow_html=True)

    df_h = get_history_df()
    if df_h is None:
        st.info("Aucune prédiction enregistrée. Rendez-vous dans Prediction Manuelle.")
        return

    n = len(df_h)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card(str(n),                                      "Total",      "#5c7a9e"),           unsafe_allow_html=True)
    c2.markdown(kpi_card(str((df_h["Prediction"] == "Low").sum()),    " Faible",   CLASS_COLORS["Low"]),  unsafe_allow_html=True)
    c3.markdown(kpi_card(str((df_h["Prediction"] == "Medium").sum()), " Modéré",   CLASS_COLORS["Medium"]), unsafe_allow_html=True)
    c4.markdown(kpi_card(str((df_h["Prediction"] == "High").sum()),   " Élevé",    CLASS_COLORS["High"]),  unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if n >= 2:
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(build_history_trend(df_h),
                            use_container_width=True, config=PLOTLY_CFG)
        with ch2:
            st.plotly_chart(build_history_distribution(df_h),
                            use_container_width=True, config=PLOTLY_CFG)

    section_header(" Détail")
    avail = [c for c in HISTORY_DISPLAY_COLS if c in df_h.columns]
    st.dataframe(df_h[avail], use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_dl, col_cl, _ = st.columns([2, 2, 4])
    with col_dl:
        st.download_button(
            "Exporter en CSV",
            data=df_to_csv_bytes(df_h),
            file_name=f"historique_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", type="primary",
        )
    with col_cl:
        if st.button(" Effacer l'historique", type="secondary"):
            st.session_state.prediction_history = []
            st.rerun()


# =============================================================================
# HELPER
# =============================================================================

def _param_desc(key: str) -> str:
    desc = {
        "n_estimators":     "Nombre d'arbres de gradient boosting",
        "max_depth":        "Profondeur maximale de chaque arbre",
        "learning_rate":    "Taux d'apprentissage (eta)",
        "subsample":        "Fraction des données utilisées par arbre",
        "colsample_bytree": "Fraction des variables utilisées par arbre",
        "min_child_weight": "Poids minimum dans un nœud feuille",
        "gamma":            "Gain minimum requis pour effectuer un split",
        "reg_alpha":        "Régularisation L1 (sparsité)",
        "reg_lambda":       "Régularisation L2 (lissage)",
        "objective":        "Fonction objectif multiclasses",
        "num_class":        "Nombre de classes cibles",
        "random_state":     "Graine aléatoire (reproductibilité)",
    }
    return desc.get(key, "")


# =============================================================================
# POINT D'ENTREE
# =============================================================================

def main():
    inject_css()
    init_session()

    try:
        model, feature_names = load_model()
    except FileNotFoundError:
        st.error(
            "Modèle introuvable.\n\n"
            "Copiez `irrigation_model.pkl` et `feature_names.pkl` dans le dossier `models/` :\n"
            "```\ncp irrigation_model.pkl  models/\ncp feature_names.pkl     models/\n```"
        )
        st.stop()
    except Exception as exc:
        st.error(f"Erreur critique : {exc}")
        st.stop()

    render_sidebar()
    page = st.session_state.current_page

    if page == "manual":
        page_manual(model, feature_names)
    elif page == "smart":
        page_smart(model, feature_names)
    elif page == "csv":
        page_csv(model, feature_names)
    elif page == "history":
        page_history()


if __name__ == "__main__":
    main()