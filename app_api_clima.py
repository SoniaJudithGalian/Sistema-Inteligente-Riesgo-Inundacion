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
# 1. CONFIGURACION Y ESTADO INICIAL
# =====================================================

st.set_page_config(page_title="Riesgo de Inundaciones", layout="wide")

# Inicialización de estado para evitar errores de Attribute Error
if "modo_bahia" not in st.session_state:
    st.session_state.modo_bahia = False

st.markdown("""
<style>
.main-title { font-size: 40px; font-weight: 800; color: #0F172A; }
.section-title { font-size: 24px; font-weight: 700; color: #0F766E; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# ... [MANTEN AQUÍ TU DICCIONARIO COORDENADAS IGUAL QUE ANTES] ...
COORDENADAS = {
    "Neuquén Capital": {"lat": -38.9516, "lon": -68.0591, "prov": "Neuquén"},
    "Plottier": {"lat": -38.9672, "lon": -68.2292, "prov": "Neuquén"},
    "Centenario": {"lat": -38.8286, "lon": -68.1436, "prov": "Neuquén"},
    "Cipolletti": {"lat": -38.9406, "lon": -67.9902, "prov": "Río Negro"},
    "General Roca": {"lat": -39.0333, "lon": -67.5833, "prov": "Río Negro"},
}

# =====================================================
# 3. CARGA Y API
# =====================================================

try:
    modelo = joblib.load("modelo_ml.pkl")
except:
    modelo = None

@st.cache_data(ttl=1800)
def obtener_clima_open_meteo(latitud, longitud):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": latitud, "longitude": longitud, "hourly": "precipitation,relative_humidity_2m", "forecast_days": 1}
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    df_clima = pd.DataFrame(data["hourly"])
    return df_clima["precipitation"].sum(), df_clima["relative_humidity_2m"].mean(), df_clima

# =====================================================
# 5. SIDEBAR
# =====================================================

st.sidebar.markdown("## Monitoreo")
ciudad_select = st.sidebar.selectbox("Ciudad", list(COORDENADAS.keys()))
datos = COORDENADAS[ciudad_select]
latitud, longitud = datos["lat"], datos["lon"]

lluvia = st.sidebar.slider("Lluvia (mm)", 0.0, 300.0, 50.0)
humedad = st.sidebar.slider("Humedad (%)", 0.0, 100.0, 50.0)
drenaje = st.sidebar.slider("Drenaje (%)", 0, 100, 50)
pendiente = st.sidebar.slider("Pendiente (%)", 0, 10, 5)

# =====================================================
# 7. INTERFAZ: RESULTADOS (Corregido para no cortar texto)
# =====================================================

st.markdown('<div class="main-title">Sistema de Riesgo</div>', unsafe_allow_html=True)

# Lógica difusa simplificada para el ejemplo
riesgo_final = (lluvia * 0.5) + (humedad * 0.2) # Ajusta según tu lógica real

col1, col2 = st.columns(2)
with col1:
    st.metric("Ciudad", ciudad_select)
with col2:
    st.metric("Nivel de Riesgo", "Alto" if riesgo_final > 50 else "Bajo")

# =====================================================
# 8. MAPA (ESTABLE Y SIN TOKENS)
# =====================================================

st.markdown('<div class="section-title">Análisis Territorial</div>', unsafe_allow_html=True)
fig_mapa = go.Figure(go.Scattermapbox(lat=[latitud], lon=[longitud], mode='markers', 
                                     marker=dict(size=20, color='red')))
fig_mapa.update_layout(mapbox=dict(style="carto-positron", center=dict(lat=latitud, lon=longitud), zoom=10),
                       margin=dict(l=0, r=0, t=0, b=0), height=350)
st.plotly_chart(fig_mapa, use_container_width=True)
