import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

import skfuzzy as fuzz
from skfuzzy import control as ctrl

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

import skfuzzy as fuzz
from skfuzzy import control as ctrl

import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# CONFIGURACION
# =====================================================

st.set_page_config(
    page_title="Riesgo de Inundaciones",
    layout="wide"
)

st.success("✅ La app inició correctamente")


# =====================================================
# CARGAR MODELO
# =====================================================

try:
    modelo = joblib.load("modelo_ml.pkl")
    st.success("✅ Modelo ML cargado correctamente")
except Exception as e:
    st.error("❌ Error al cargar el modelo ML")
    st.exception(e)
    st.stop()


# =====================================================
# API DE CLIMA - OPEN METEO
# =====================================================

@st.cache_data(ttl=1800)
def obtener_clima_open_meteo(latitud, longitud):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitud,
        "longitude": longitud,
        "hourly": "precipitation,relative_humidity_2m",
        "forecast_days": 1,
        "timezone": "America/Argentina/Buenos_Aires"
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    df_clima = pd.DataFrame(data["hourly"])

    lluvia_24h = df_clima["precipitation"].sum()
    humedad_promedio = df_clima["relative_humidity_2m"].mean()

    return lluvia_24h, humedad_promedio, df_clima
