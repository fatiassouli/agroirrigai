# 🌱 AgroIrrigAI

> **Machine Learning platform for intelligent irrigation decision support**

AgroIrrigAI is a Streamlit application developed as part of a **Master's degree in Artificial Intelligence**. It uses an **XGBoost classification model** to estimate the irrigation requirement of an agricultural plot as **Low, Medium, or High**.

The application combines agronomic parameters, environmental conditions and real-time weather data to provide an interactive decision-support interface.

## ✨ Key Features

- 🔬 **Manual ML prediction** from agronomic and environmental parameters
- 🌦️ **Weather-based prediction** using OpenWeather
- 📂 **Batch prediction** from CSV files
- 📊 **Prediction probabilities** and risk visualizations with Plotly
- 🏆 **XGBoost feature importance**
- 📜 **Session prediction history**
- 📥 **CSV export** of prediction results
- 🛡️ Input validation and secure API-key configuration
- 🎨 Responsive Streamlit interface with a dedicated agricultural theme

## 🧠 Machine Learning

The prediction pipeline reproduces the preprocessing used during model training:

1. Ordinal encoding of crop growth stage
2. Binary encoding of mulching
3. One-hot encoding of categorical variables
4. Boolean-to-integer conversion
5. Feature alignment with the training schema
6. XGBoost inference
7. Custom safety rule: if `P(High) > 0.60`, the prediction is forced to `High`

### Output classes

| Class | Interpretation |
|---|---|
| 💧 Low | Low irrigation requirement |
| 🌿 Medium | Moderate irrigation requirement |
| 🔥 High | High irrigation requirement |

## 🏗️ Project Architecture

```text
agroirrigai/
│
├── app.py                         # Streamlit application
├── prediction.py                  # ML preprocessing and inference
├── weather_service.py             # OpenWeather integration
├── cities.py                      # City list
├── config.py                      # Application and ML configuration
├── utils.py                       # Charts, history and exports
│
├── models/
│   ├── irrigation_model.pkl       # Trained XGBoost model
│   └── feature_names.pkl          # Training feature schema
│
├── .streamlit/
│   ├── config.toml                # Streamlit theme/server config
│   └── secrets.toml.example       # Secret configuration template
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/fatiassouli/agroirrigai.git
cd agroirrigai
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure OpenWeather

Create:

```text
.streamlit/secrets.toml
```

with:

```toml
OPENWEATHER_API_KEY = "YOUR_OPENWEATHER_API_KEY"
```

The file is intentionally ignored by Git and must **never** be committed.

### 5. Start the application

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## ☁️ Deployment

The application is suitable for deployment on **Streamlit Community Cloud**.

Typical deployment flow:

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Install requirements.txt
       ↓
Configure OPENWEATHER_API_KEY
       ↓
Run app.py
```

For Streamlit Cloud, add the API key under the application's **Secrets** settings instead of committing `.streamlit/secrets.toml`.

## 🔐 Security

API credentials are deliberately excluded from version control.

- `.streamlit/secrets.toml` is ignored by Git
- `secrets.toml.example` contains only a placeholder
- OpenWeather credentials should be stored as deployment secrets
- Never paste API keys directly into Python source files

> **Important:** if an API key has ever been committed to a public repository, revoke/rotate it from the provider dashboard and replace it with a new key.

## 📌 Project Status

**Status:** Academic / portfolio project — functional prototype.

The project demonstrates the integration of:

**Machine Learning · XGBoost · Python · Streamlit · Data Processing · Data Visualization · REST API Integration**

## 👩‍💻 Author

**Fatima Zahraa Assouli**  
Master's student in Artificial Intelligence — Faculté des Sciences de Ben M'Sik, Casablanca, Morocco

- LinkedIn: https://www.linkedin.com/in/fatima-zahraa-assouli-204008279/
- GitHub: https://github.com/fatiassouli

## 📄 License

MIT License.
