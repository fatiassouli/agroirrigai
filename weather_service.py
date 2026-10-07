import requests
import streamlit as st


def get_weather(city: str):

    api_key = st.secrets["OPENWEATHER_API_KEY"]

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}"
        f"&appid={api_key}"
        "&units=metric"
    )

    response = requests.get(url, timeout=10)
    data = response.json()

    sunrise = data["sys"]["sunrise"]
    sunset = data["sys"]["sunset"]

    return {
        "Temperature_C": data["main"]["temp"],
        "Humidity": data["main"]["humidity"],
        "Rainfall_mm": data.get("rain", {}).get("1h", 0),
        "Wind_Speed_kmh": data["wind"]["speed"] * 3.6,
        "Sunlight_Hours": round((sunset - sunrise) / 3600, 1),
    }