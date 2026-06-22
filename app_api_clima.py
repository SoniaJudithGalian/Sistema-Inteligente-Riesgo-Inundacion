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
# 1. CONFIGURACION DE LA PÁGINA Y ESTILOS
# =====================================================

st.set_page_config(page_title="Riesgo de Inundaciones", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-title { font-size: 46px; font-weight: 800; color: #0F172A; margin-bottom: 0px; }
.subtitle { font-size: 18px; color: #475569; margin-bottom: 25px; }
.section-title { font-size: 28px; font-weight: 700; color: #0F766E; margin-top: 25px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. BASE DE DATOS GEOGRÁFICA
# =====================================================

COORDENADAS = {
    "Neuquén Capital": {"lat": -38.9516, "lon": -68.0591, "prov": "Neuquén"},
    "Plottier": {"lat": -38.9672, "lon": -68.2292, "prov": "Neuquén"},
    "Centenario": {"lat": -38.8286, "lon": -68.1436, "prov": "Neuquén"},
    "Añelo": {"lat": -38.3517, "lon": -68.7881, "prov": "Neuquén"},
    "Cipolletti": {"lat": -38.9406, "lon": -67.9902, "prov": "Río Negro"},
    "General Roca": {"lat": -39.0333, "lon": -67.5833, "prov": "Río Negro"},
    "La Plata": {"lat": -34.9215, "lon": -57.9545, "prov": "Buenos Aires"},
    "Santa Fe Capital": {"lat": -31.6333, "lon": -60.7000, "prov": "Santa Fe"},
    "Resistencia": {"lat": -27.4606, "lon": -58.9839, "prov": "Chaco"},
    "Comodoro Rivadavia": {"lat": -45.8641, "lon": -67.4966, "prov": "Chubut"},
    "Concordia": {"lat": -31.3930, "lon": -58.0209, "prov": "Entre Ríos"},
    "Buenos Aires": {"lat": -34.6037, "lon": -58.3816, "prov": "CABA"},
    "Córdoba": {"lat": -31.4135, "lon": -64.1811, "prov": "Córdoba"},
    "Rosario": {"lat": -32.9468, "lon": -60.6393, "prov": "Santa Fe"},
    "Salta": {"lat": -24.7821, "lon": -65.4233, "prov": "Salta"},
    "Mendoza": {"lat": -32.8908, "lon": -68.8272, "prov": "Mendoza"}
}

# =====================================================
# 3. CARGAR MODELOS Y APIS
# =====================================================

try:
    modelo = joblib.load("modelo_ml.pkl")
except Exception as e:
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
# 4. TÍTULO PRINCIPAL
# =====================================================

st.markdown('<div class="main-title">Sistema Inteligente de Riesgo de Inundaciones</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Aplicación con Machine Learning, Lógica Difusa y datos climáticos.</div>', unsafe_allow_html=True)

# =====================================================
# 5. SIDEBAR
# =====================================================

st.sidebar.markdown("## Monitoreo territorial")
opciones_ciudades = list(COORDENADAS.keys()) + ["Bahía Blanca (Caso Histórico 2025)"]
ciudad_select = st.sidebar.selectbox("Elegí una ciudad", opciones_ciudades)

df_clima = None
es_caso_historico = False

if ciudad_select == "Bahía Blanca (Caso Histórico 2025)":
    es_caso_historico = True
    ciudad, provincia = "Bahía Blanca", "Buenos Aires"
    latitud, longitud = -38.7196, -62.2724
    lluvia, humedad = 290.0, 95.0
    drenaje, pendiente = 30, 2
else:
    ciudad = ciudad_select
    latitud, longitud = COORDENADAS[ciudad_select]["lat"], COORDENADAS[ciudad_select]["lon"]
    lluvia, humedad, df_clima = obtener_clima_open_meteo(latitud, longitud)
    drenaje = st.sidebar.slider("Drenaje (%)", 0, 100, 50)
    pendiente = st.sidebar.slider("Pendiente (%)", 0, 10, 5)

# =====================================================
# 6. LÓGICA Y RESULTADOS
# =====================================================

prediccion_texto = "Baja"
if modelo:
    nuevo_caso = pd.DataFrame({"lluvia_mm": [lluvia], "humedad_suelo": [humedad], "capacidad_drenaje": [drenaje], "pendiente_topografica": [pendiente]})
    prediccion_texto = "Alto" if modelo.predict(nuevo_caso)[0] > 0 else "Bajo"

nivel_difuso = "Bajo" if lluvia < 50 else "Alto"
riesgo_final = 20 if lluvia < 50 else 70

st.markdown('<div class="section-title">Resumen de Situación</div>', unsafe_allow_html=True)
with st.container(border=True):
    col1, col2 = st.columns(2)
    col1.metric("Ciudad", ciudad)
    col1.metric("Modelo ML", prediccion_texto)
    col2.metric("Lluvia", f"{round(lluvia, 1)} mm")
    col2.metric("Sistema Difuso", f"{nivel_difuso} ({riesgo_final} / 100)")

# =====================================================
# 8. MAPA ESTABLE (PLOTLY CON PUNTO)
# =====================================================

st.markdown('<div class="section-title">Análisis Territorial</div>', unsafe_allow_html=True)

fig_mapa = go.Figure()
fig_mapa.add_trace(go.Scattermapbox(
    lat=[latitud], lon=[longitud], mode='markers',
    marker=dict(size=25, color='red' if riesgo_final > 50 else 'green', opacity=0.8),
    text=[ciudad]
))

fig_mapa.update_layout(
    mapbox=dict(style="carto-positron", center=dict(lat=latitud, lon=longitud), zoom=10),
    margin=dict(l=0, r=0, t=0, b=0), height=380
)
st.plotly_chart(fig_mapa, use_container_width=True)
