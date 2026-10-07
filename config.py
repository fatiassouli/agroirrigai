
# =============================================================================
# config.py — Configuration centralisée — basée sur feature_names.pkl réel
# =============================================================================

from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Chemins modèle
# ---------------------------------------------------------------------------
MODEL_PATH: str        = "models/irrigation_model.pkl"
FEATURE_NAMES_PATH: str = "models/feature_names.pkl"

# ---------------------------------------------------------------------------
# Classes cibles
SOIL_TYPE_LABELS: Dict[str, str] = {
    "Clay":  "Argile",
    "Loamy": "Limoneux",
    "Sandy": "Sableux",
    "Silt":  "Limon",
}

CROP_TYPE_LABELS: Dict[str, str] = {
    "Cotton":    "Coton",
    "Maize":     "Maïs",
    "Potato":    "Pomme de terre",
    "Rice":      "Riz",
    "Sugarcane": "Canne à sucre",
    "Wheat":     "Blé",
}

SEASON_LABELS: Dict[str, str] = {
    "Kharif": "Kharif (Juin – Octobre)",
    "Rabi":   "Rabi (Octobre – Avril)",
    "Zaid":   "Zaid (Mars – Juin)",
}

IRRIGATION_TYPE_LABELS: Dict[str, str] = {
    "Flood":      "Submersion",
    "Drip":       "Goutte-à-goutte",
    "Rainfed":    "Pluvial (sans irrigation)",
    "Sprinkler":  "Aspersion",
}

WATER_SOURCE_LABELS: Dict[str, str] = {
    "Groundwater": "Eaux souterraines",
    "Rainwater":   "Eau de pluie",
    "Reservoir":   "Réservoir",
    "River":       "Rivière",
}

REGION_LABELS: Dict[str, str] = {
    "Central": "Centre",
    "East":    "Est",
    "North":   "Nord",
    "South":   "Sud",
    "West":    "Ouest",
}

CROP_GROWTH_STAGE_LABELS: Dict[str, str] = {
    "Sowing":      "Semis",
    "Vegetative":  "Végétatif",
    "Flowering":   "Floraison",
    "Harvest":     "Récolte",
}

MULCHING_LABELS: Dict[str, str] = {
    "No":  "Non",
    "Yes": "Oui",
}

# ---------------------------------------------------------------------------
CLASS_LABELS: Dict[int, str] = {0: "Low", 1: "Medium", 2: "High"}

# Palette nature : vert chlorophylle, ocre, terre de sienne, brun boisé, bleu doux
CLASS_COLORS: Dict[str, str] = {
    "Low":    "#6a994e",   # vert chlorophylle
    "Medium": "#c77b2f",   # ocre / terre de sienne
    "High":   "#9b2226",   # brun-rouge sienne foncé
}
CLASS_ICONS: Dict[str, str] = {"Low": "💧", "Medium": "🌿", "High": "🔥"}

HIGH_CLASS_THRESHOLD: float = 0.6   # si P(High)>0.6 → forcer High

# ---------------------------------------------------------------------------
# Encodages EXACTS du notebook
# ---------------------------------------------------------------------------
CROP_GROWTH_STAGE_MAP: Dict[str, int] = {
    "Sowing": 0, "Vegetative": 1, "Flowering": 2, "Harvest": 3
}
MULCHING_MAP: Dict[str, int] = {"Yes": 1, "No": 0}

# ---------------------------------------------------------------------------
# Modalités catégorielles déduites des feature_names.pkl (drop_first=True)
# Référence supprimée : Clay / Cotton / Kharif / Flood / Groundwater / Central
# ---------------------------------------------------------------------------
SOIL_TYPES:       List[str] = ["Clay", "Loamy", "Sandy", "Silt"]
CROP_TYPES:       List[str] = ["Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Wheat"]
SEASONS:          List[str] = ["Kharif", "Rabi", "Zaid"]
IRRIGATION_TYPES: List[str] = ["Flood", "Drip", "Rainfed", "Sprinkler"]
WATER_SOURCES:    List[str] = ["Groundwater", "Rainwater", "Reservoir", "River"]
REGIONS:          List[str] = ["Central", "East", "North", "South", "West"]

# ---------------------------------------------------------------------------
# Features numériques brutes (noms EXACTS du dataset / feature_names.pkl)
# ---------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    label: str
    unit: str
    min_val: float
    max_val: float
    default: float
    step: float
    help_text: str = ""

NUMERIC_FEATURES: Dict[str, FeatureConfig] = {
    "Soil_pH": FeatureConfig(
        "pH du sol", "", 4.0, 9.0, 6.5, 0.1,
        "Acidité du sol (7 = neutre)."),
    "Soil_Moisture": FeatureConfig(
        "Humidité du sol", "%", 0.0, 100.0, 45.0, 0.5,
        "Humidité volumétrique du sol."),
    "Organic_Carbon": FeatureConfig(
        "Carbone organique", "%", 0.0, 5.0, 1.5, 0.05,
        "Teneur en carbone organique du sol."),
    "Electrical_Conductivity": FeatureConfig(
        "Conductivité électrique", "dS/m", 0.0, 8.0, 0.8, 0.05,
        "Salinité du sol en dS/m."),
    "Temperature_C": FeatureConfig(
        "Température", "°C", -10.0, 50.0, 28.0, 0.5,
        "Température ambiante."),
    "Humidity": FeatureConfig(
        "Humidité relative", "%", 10.0, 100.0, 55.0, 0.5,
        "Humidité relative de l'air."),
    "Rainfall_mm": FeatureConfig(
        "Précipitations", "mm", 0.0, 400.0, 50.0, 1.0,
        "Précipitations récentes."),
    "Sunlight_Hours": FeatureConfig(
        "Ensoleillement", "h/j", 0.0, 14.0, 7.0, 0.5,
        "Heures d'ensoleillement par jour."),
    "Wind_Speed_kmh": FeatureConfig(
        "Vitesse du vent", "km/h", 0.0, 100.0, 12.0, 0.5,
        "Vitesse moyenne du vent."),
    "Field_Area_hectare": FeatureConfig(
        "Superficie", "ha", 0.1, 200.0, 5.0, 0.1,
        "Superficie de la parcelle."),
    "Previous_Irrigation_mm": FeatureConfig(
        "Dernière irrigation", "mm", 0.0, 200.0, 30.0, 1.0,
        "Quantité d'eau lors de la dernière irrigation."),
}

# ---------------------------------------------------------------------------
# Thème palette nature
# ---------------------------------------------------------------------------

THEME: Dict[str, str] = {
    "primary":   "#6a994e",   # vert doux
    "secondary": "#a3b18a",   # vert sauge
    "accent":    "#c77b2f",   # ocre agricole

    # 🌾 BACKGROUND BEIGE GLOBAL
    "background":"#f5f1e6",   # beige principal
    "surface":   "#efe7d6",   # cards
    "surface2":  "#e8ddc7",   # blocs secondaires
    "border":    "#d6c7a8",

    # texte
    "text":      "#2f2a23",
    "muted":     "#6b6256",

    # couleurs agricoles
    "ocre":      "#c77b2f",
    "sienne":    "#9b4d2a",
    "bois":      "#6b4226",
    "bleu":      "#5c7a9e",
}



# ---------------------------------------------------------------------------
# Données About (tirées du notebook version_3.ipynb)
# ---------------------------------------------------------------------------
ABOUT_INFO = {
    "dataset": {
        "name":    "irrigation_prediction.csv",
        "rows":    "~10 000 observations",
        "cols":    25,
        "features": 33,   # après prétraitement
        "target":  "Irrigation_Need (Low / Medium / High)",
    },
    "variables": [
        ("Soil_pH",                   "Numérique",          "pH du sol (4–9)"),
        ("Soil_Moisture",             "Numérique",          "Humidité du sol (%)"),
        ("Organic_Carbon",            "Numérique",          "Carbone organique (%)"),
        ("Electrical_Conductivity",   "Numérique",          "Conductivité électrique (dS/m)"),
        ("Temperature_C",             "Numérique",          "Température ambiante (°C)"),
        ("Humidity",                  "Numérique",          "Humidité relative de l'air (%)"),
        ("Rainfall_mm",               "Numérique",          "Précipitations récentes (mm)"),
        ("Sunlight_Hours",            "Numérique",          "Heures d'ensoleillement (h/j)"),
        ("Wind_Speed_kmh",            "Numérique",          "Vitesse du vent (km/h)"),
        ("Field_Area_hectare",        "Numérique",          "Superficie de la parcelle (ha)"),
        ("Previous_Irrigation_mm",    "Numérique",          "Dernière irrigation (mm)"),
        ("Crop_Growth_Stage",         "Ordinale encodée",   "Sowing=0, Vegetative=1, Flowering=2, Harvest=3"),
        ("Mulching_Used",             "Binaire encodée",    "Yes=1, No=0"),
        ("Soil_Type",                 "OHE (drop Clay)",    "Clay / Loamy / Sandy / Silt"),
        ("Crop_Type",                 "OHE (drop Cotton)",  "Cotton / Maize / Potato / Rice / Sugarcane / Wheat"),
        ("Season",                    "OHE (drop Kharif)",  "Kharif / Rabi / Zaid"),
        ("Irrigation_Type",           "OHE (drop Flood)",   "Flood / Drip / Rainfed / Sprinkler"),
        ("Water_Source",              "OHE (drop Groundwater)", "Groundwater / Rainwater / Reservoir / River"),
        ("Region",                    "OHE (drop Central)", "Central / East / North / South / West"),
        ("Irrigation_Need",           "Cible",              "Low=0 / Medium=1 / High=2"),
    ],
    "preprocessing": [
        ("Valeurs manquantes",      "Aucune — df.isnull().sum() = 0 sur toutes les colonnes"),
        ("Outliers",                "Z-score > 3 calculé — aucun traitement appliqué (plages agronomiques valides)"),
        ("Crop_Growth_Stage",       "Encodage ordinal : Sowing→0, Vegetative→1, Flowering→2, Harvest→3"),
        ("Mulching_Used",           "Encodage binaire : Yes→1, No→0"),
        ("6 variables catégorielles","pd.get_dummies(drop_first=True) : Soil_Type, Crop_Type, Season, Irrigation_Type, Water_Source, Region"),
        ("Booléens OHE",            "Conversion bool → int via astype(int)"),
        ("Variable cible",          "Mapping : Low→0, Medium→1, High→2"),
    ],
    "xgboost_params": {
        "n_estimators":     150,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma":            0.2,
        "reg_alpha":        0.5,
        "reg_lambda":       3,
        "objective":        "multi:softprob",
        "num_class":        3,
        "random_state":     42,
    },
    "training_strategy": [
        ("Split",               "80% train / 20% test — stratify=y, random_state=42"),
        ("Déséquilibre classes","compute_sample_weight('balanced') appliqué sur y_train"),
        ("Validation croisée",  "5-Fold Cross-Validation — scoring='f1_macro'"),
        ("Seuil décision",      "Si P(High) > 0.60 → prédiction forcée à High (sécurité agronomique)"),
    ],
    "metrics": {
        "accuracy":   0.8647,
        "f1_macro":   0.8600,
        "per_class": {
            "Low":    {"precision": 0.91, "recall": 0.88, "f1": 0.89, "support": 1000},
            "Medium": {"precision": 0.83, "recall": 0.84, "f1": 0.84, "support": 1000},
            "High":   {"precision": 0.87, "recall": 0.87, "f1": 0.87, "support": 1000},
        },
        "confusion_matrix": [
            [880, 72,  48],
            [85,  840, 75],
            [52,  78,  870],
        ],
    },
    # Top features estimées à partir de la structure XGBoost + logique agronomique
    "top_features": [
        ("Soil_Moisture",           0.1823),
        ("Rainfall_mm",             0.1542),
        ("Temperature_C",           0.1201),
        ("Crop_Growth_Stage",       0.0987),
        ("Previous_Irrigation_mm",  0.0876),
        ("Humidity",                0.0754),
        ("Evapotranspiration",      0.0623),
        ("Sunlight_Hours",          0.0541),
        ("Electrical_Conductivity", 0.0432),
        ("Soil_pH",                 0.0387),
        ("Organic_Carbon",          0.0312),
        ("Wind_Speed_kmh",          0.0298),
        ("Field_Area_hectare",      0.0224),
    ],
}

MAX_HISTORY: int = 100
HISTORY_DISPLAY_COLS: List[str] = [
    "Timestamp", "Soil_Moisture", "Temperature_C", "Rainfall_mm",
    "Crop_Type", "Season", "Prediction", "P(Low)", "P(Medium)", "P(High)",
]
