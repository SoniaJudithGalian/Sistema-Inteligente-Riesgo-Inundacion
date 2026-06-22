import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

# =====================================================
# 1. CONFIGURACION Y ESTILO
# =====================================================
st.set_page_config(page_title="Riesgo de Inundaciones", layout="wide")

st.markdown("""
<style>
.main-title { font-size: 46px; font-weight: 800; color: #0F172A; }
.section-title { font-size: 28px; font-weight: 700; color: #0F766E; margin-top: 25px; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. BASE DE DATOS
# =====================================================
COORDENADAS = {
    "Neuquén Capital": {"lat": -38.9516, "lon": -68.0591, "prov": "Neuquén"},
    "Plottier": {"lat": -38.9672, "lon": -68.2292, "prov": "Neuquén"},
    "Centenario": {"lat": -38.8286, "lon": -68.1436, "prov": "Neuquén"},
    "Añelo": {"lat": -38.3517, "lon": -68.7881, "prov": "Neuquén"},
    "Cipolletti": {"lat": -38.9406, "lon": -67.9902, "prov": "Río Negro"},
    "General Roca": {"lat": -39.0333, "lon": -67.5833, "prov": "Río Negro"},
    "Mendoza": {"lat": -32.8908, "lon": -68.8272, "prov": "Mendoza"}
}

# =====================================================
# 3. CARGA DE MODELO Y CLIMA
# =====================================================
try:
    modelo = joblib.load("modelo_ml.pkl")
except:
    modelo = None

@st.cache_data(ttl=1800)
def obtener_clima_open_meteo(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation,relative_humidity_2m&forecast_days=1"
    res = requests.get(url).json()
    df = pd.DataFrame(res["hourly"])
    return df["precipitation"].sum(), df["relative_humidity_2m"].mean(), df

# =====================================================
# 4. INTERFAZ Y SIDEBAR
# =====================================================
st.markdown('<div class="main-title">Sistema Inteligente de Riesgo de Inundaciones</div>', unsafe_allow_html=True)
ciudad_select = st.sidebar.selectbox("Ciudad", list(COORDENADAS.keys()))
lat, lon = COORDENADAS[ciudad_select]["lat"], COORDENADAS[ciudad_select]["lon"]
lluvia, humedad, df_clima = obtener_clima_open_meteo(lat, lon)

# Lógica básica de riesgo para los gráficos
riesgo_final = 10 if lluvia < 20 else 75
nivel_difuso = "Bajo" if riesgo_final < 50 else "Alto"

# =====================================================
# 5. RESULTADOS (Sin emojis)
# =====================================================
st.markdown('<div class="section-title">Resumen de Situación</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
c1.metric("Ciudad", ciudad_select)
c1.metric("Nivel de Riesgo", nivel_difuso)
c2.metric("Lluvia", f"{round(lluvia, 1)} mm")
c2.metric("Sistema Difuso", f"{riesgo_final} / 100")

# =====================================================
# 6. MAPA SATELITAL CON PUNTO DE UBICACIÓN (Pydeck)
# =====================================================
st.markdown('<div class="section-title">Análisis Territorial</div>', unsafe_allow_html=True)

# Datos para el punto
data = pd.DataFrame({'lat': [lat], 'lon': [lon]})

# Capa Satelital + Punto de ubicación
st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=11, pitch=0),
    layers=[
        pdk.Layer("TileLayer", "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        pdk.Layer("ScatterplotLayer", data, get_position="[lon, lat]", get_fill_color=[255, 0, 0, 200], get_radius=1000)
    ]
))

# =====================================================
# 7. GRÁFICOS
# =====================================================
st.markdown('<div class="section-title">Desglose de Datos</div>', unsafe_allow_html=True)
fig = px.bar(df_clima, x="time", y="precipitation", title=f"Pronóstico {ciudad_select}")
st.plotly_chart(fig, use_container_width=True)
