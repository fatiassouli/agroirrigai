# =============================================================================
# prediction.py — Moteur de prétraitement et d'inférence
# Pipeline EXACT du notebook version_3.ipynb
# =============================================================================

import logging
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    CLASS_LABELS, CROP_GROWTH_STAGE_MAP, FEATURE_NAMES_PATH,
    HIGH_CLASS_THRESHOLD, MODEL_PATH, MULCHING_MAP,
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="⏳ Chargement du modèle XGBoost…")
def load_model() -> Tuple[Any, List[str]]:
    """Charge le modèle et les feature names depuis models/."""
    try:
        model = joblib.load(MODEL_PATH)
        feature_names: List[str] = joblib.load(FEATURE_NAMES_PATH)
        logger.info("Modèle chargé — %d features.", len(feature_names))
        return model, feature_names
    except FileNotFoundError as exc:
        logger.error("Fichier introuvable : %s", exc)
        raise


# ---------------------------------------------------------------------------
# Prétraitement — reproduction exacte du notebook
# ---------------------------------------------------------------------------

def _ordinal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Crop_Growth_Stage : Sowing=0, Vegetative=1, Flowering=2, Harvest=3"""
    df = df.copy()
    if "Crop_Growth_Stage" in df.columns:
        df["Crop_Growth_Stage"] = df["Crop_Growth_Stage"].map(CROP_GROWTH_STAGE_MAP)
    return df


def _binary_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Mulching_Used : Yes=1, No=0"""
    df = df.copy()
    if "Mulching_Used" in df.columns:
        df["Mulching_Used"] = df["Mulching_Used"].map(MULCHING_MAP)
    return df


def _ohe(df: pd.DataFrame) -> pd.DataFrame:
    """OHE avec drop_first=True — identique à pd.get_dummies du notebook."""
    categorical_cols = [
        "Soil_Type", "Crop_Type", "Season",
        "Irrigation_Type", "Water_Source", "Region",
    ]
    cols_present = [c for c in categorical_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cols_present, drop_first=True)
    return df


def _cast_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """bool → int (identique notebook)."""
    df = df.copy()
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def _align(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Aligne sur les colonnes d'entraînement — manquantes = 0."""
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


def preprocess(raw_input: Dict[str, Any], feature_names: List[str]) -> pd.DataFrame:
    """Pipeline complet : ordinal → binaire → OHE → bool→int → alignement."""
    df = pd.DataFrame([raw_input])
    df = _ordinal_encoding(df)
    df = _binary_encoding(df)
    df = _ohe(df)
    df = _cast_booleans(df)
    df = _align(df, feature_names)
    return df


# ---------------------------------------------------------------------------
# Inférence
# ---------------------------------------------------------------------------

def predict(model: Any, feature_names: List[str], raw_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prédit le besoin d'irrigation.
    Seuil personnalisé : P(High) > 0.6 → forcer High (logique du notebook).
    """
    X = preprocess(raw_input, feature_names)
    proba: np.ndarray = model.predict_proba(X)[0]

    if proba[2] > HIGH_CLASS_THRESHOLD:
        pred_int = 2
    else:
        pred_int = int(np.argmax(proba))

    return {
        "prediction_label": CLASS_LABELS[pred_int],
        "prediction_int":   pred_int,
        "probabilities":    {"Low": float(proba[0]), "Medium": float(proba[1]), "High": float(proba[2])},
        "confidence":       float(proba[pred_int]),
    }


def predict_batch(model: Any, feature_names: List[str], df_raw: pd.DataFrame) -> pd.DataFrame:
    """Prédictions sur un DataFrame entier (upload CSV)."""
    results = []
    for _, row in df_raw.iterrows():
        try:
            res = predict(model, feature_names, row.to_dict())
            results.append({
                "Prediction": res["prediction_label"],
                "P(Low)":     round(res["probabilities"]["Low"],    4),
                "P(Medium)":  round(res["probabilities"]["Medium"],  4),
                "P(High)":    round(res["probabilities"]["High"],    4),
                "Confidence": round(res["confidence"],               4),
            })
        except Exception as exc:
            logger.warning("Ligne ignorée : %s", exc)
            results.append({"Prediction": "ERROR", "P(Low)": None, "P(Medium)": None, "P(High)": None, "Confidence": None})

    return pd.concat([df_raw.reset_index(drop=True), pd.DataFrame(results)], axis=1)


def get_feature_importance(model: Any, feature_names: List[str], top_n: int = 15) -> pd.DataFrame:
    """Retourne le DataFrame des feature importances trié."""
    df = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    return df.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)
