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
    st.error("Error al cargar el modelo ML. Verifica el archivo 'modelo_ml.pkl'")
    st.stop()

@st.cache_data(ttl=1800)
def obtener_clima_open_meteo(latitud, longitud):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitud, "longitude": longitud,
        "hourly": "precipitation,relative_humidity_2m",
        "forecast_days": 1, "timezone": "America/Argentina/Buenos_Aires"
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    df_clima = pd.DataFrame(data["hourly"])
    lluvia_24h = df_clima["precipitation"].sum()
    humedad_promedio = df_clima["relative_humidity_2m"].mean()
    return lluvia_24h, humedad_promedio, df_clima

# =====================================================
# 4. TÍTULO PRINCIPAL
# =====================================================

st.markdown('<div class="main-title">Sistema Inteligente de Riesgo de Inundaciones</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Aplicación con Machine Learning, Lógica Difusa y datos climáticos de Open-Meteo.</div>', unsafe_allow_html=True)

# =====================================================
# 5. SIDEBAR Y SELECCIÓN DE DATOS
# =====================================================

st.sidebar.markdown("## Monitoreo territorial")

opciones_ciudades = list(COORDENADAS.keys()) + ["Bahía Blanca (Caso Histórico 2025)"]
ciudad_select = st.sidebar.selectbox("Elegí una ciudad para analizar", opciones_ciudades)

df_clima = None
es_caso_historico = False

if ciudad_select == "Bahía Blanca (Caso Histórico 2025)":
    es_caso_historico = True
    ciudad = "Bahía Blanca"
    provincia = "Buenos Aires"
    latitud, longitud = -38.7196, -62.2724
    lluvia, humedad = 290.0, 95.0
    drenaje, pendiente = 30, 2
    st.sidebar.warning("⚠️ Modo Simulación: Evento histórico extremo cargado.")
else:
    ciudad = ciudad_select
    datos_ubicacion = COORDENADAS[ciudad_select]
    latitud = datos_ubicacion["lat"]
    longitud = datos_ubicacion["lon"]
    provincia = datos_ubicacion["prov"]

    st.sidebar.markdown("### Datos Climáticos")
    usar_api = st.sidebar.checkbox("Usar datos climáticos automáticos", value=True)

    if usar_api:
        try:
            lluvia_api, humedad_api, df_clima = obtener_clima_open_meteo(latitud, longitud)
            lluvia = round(float(lluvia_api), 2)
            humedad = round(float(humedad_api), 2)
            st.sidebar.success("Datos actualizados")
            st.sidebar.metric("Lluvia estimada 24h", f"{lluvia} mm")
            st.sidebar.metric("Humedad promedio", f"{humedad} %")
        except:
            st.sidebar.error("Error API. Usa modo manual.")
            lluvia = st.sidebar.slider("Lluvia manual (mm)", 0.0, 300.0, 50.0)
            humedad = st.sidebar.slider("Humedad manual (%)", 0.0, 100.0, 50.0)
    else:
        lluvia = st.sidebar.slider("Lluvia manual (mm)", 0.0, 300.0, 50.0)
        humedad = st.sidebar.slider("Humedad manual (%)", 0.0, 100.0, 50.0)

    st.sidebar.markdown("### Territorio")
    drenaje = st.sidebar.slider("Capacidad de drenaje (%)", 0, 100, 50)
    pendiente = st.sidebar.slider("Pendiente topográfica (%)", 0, 10, 5)

# =====================================================
# 6. MACHINE LEARNING & LÓGICA DIFUSA
# =====================================================

nuevo_caso = pd.DataFrame({"lluvia_mm": [lluvia], "humedad_suelo": [humedad], "capacidad_drenaje": [drenaje], "pendiente_topografica": [pendiente]})
prediccion_ml = modelo.predict(nuevo_caso)

clases_riesgo = {0: "Bajo", 1: "Medio", 2: "Alto", 3: "Crítico"}
prediccion_texto = clases_riesgo.get(int(prediccion_ml[0]), "Desconocido")

lluvia_fz = ctrl.Antecedent(np.arange(0, 121, 1), "lluvia")
humedad_fz = ctrl.Antecedent(np.arange(0, 101, 1), "humedad")
drenaje_fz = ctrl.Antecedent(np.arange(0, 101, 1), "drenaje")
pendiente_fz = ctrl.Antecedent(np.arange(0, 11, 1), "pendiente")
riesgo = ctrl.Consequent(np.arange(0, 101, 1), "riesgo")

lluvia_fz["baja"] = fuzz.trimf(lluvia_fz.universe, [0, 0, 40]); lluvia_fz["media"] = fuzz.trimf(lluvia_fz.universe, [20, 60, 90]); lluvia_fz["alta"] = fuzz.trimf(lluvia_fz.universe, [70, 120, 120])
humedad_fz["seca"] = fuzz.trimf(humedad_fz.universe, [0, 0, 40]); humedad_fz["media"] = fuzz.trimf(humedad_fz.universe, [20, 50, 80]); humedad_fz["saturada"] = fuzz.trimf(humedad_fz.universe, [60, 100, 100])
drenaje_fz["bajo"] = fuzz.trimf(drenaje_fz.universe, [0, 0, 40]); drenaje_fz["medio"] = fuzz.trimf(drenaje_fz.universe, [20, 50, 80]); drenaje_fz["alto"] = fuzz.trimf(drenaje_fz.universe, [60, 100, 100])
pendiente_fz["llana"] = fuzz.trimf(pendiente_fz.universe, [0, 0, 3]); pendiente_fz["moderada"] = fuzz.trimf(pendiente_fz.universe, [2, 5, 8]); pendiente_fz["empinada"] = fuzz.trimf(pendiente_fz.universe, [6, 10, 10])
riesgo["bajo"] = fuzz.trimf(riesgo.universe, [0, 0, 40]); riesgo["medio"] = fuzz.trimf(riesgo.universe, [20, 50, 80]); riesgo["alto"] = fuzz.trimf(riesgo.universe, [60, 100, 100])

reglas = [
    ctrl.Rule(lluvia_fz["alta"] & humedad_fz["saturada"] & drenaje_fz["bajo"], riesgo["alto"]),
    ctrl.Rule(lluvia_fz["media"] & drenaje_fz["medio"], riesgo["medio"]),
    ctrl.Rule(lluvia_fz["baja"] & drenaje_fz["alto"], riesgo["bajo"]),
    ctrl.Rule(pendiente_fz["llana"] & lluvia_fz["alta"], riesgo["alto"]),
    ctrl.Rule(lluvia_fz["alta"], riesgo["alto"]),
    ctrl.Rule(lluvia_fz["media"], riesgo["medio"]),
    ctrl.Rule(lluvia_fz["baja"], riesgo["bajo"]),
    ctrl.Rule((lluvia_fz["media"] | lluvia_fz["alta"]) & drenaje_fz["bajo"], riesgo["alto"]),
    ctrl.Rule(drenaje_fz["alto"], riesgo["bajo"])
]

sistema = ctrl.ControlSystem(reglas)
simulador = ctrl.ControlSystemSimulation(sistema)
simulador.input["lluvia"] = lluvia if lluvia <= 120 else 120
simulador.input["humedad"] = humedad
simulador.input["drenaje"] = drenaje
simulador.input["pendiente"] = pendiente
simulador.compute()

riesgo_final = simulador.output.get("riesgo", 0)

def obtener_nivel_difuso(valor):
    if valor >= 80: return "Crítico"
    elif valor >= 60: return "Alto"
    elif valor >= 40: return "Medio"
    else: return "Bajo"

nivel_difuso = obtener_nivel_difuso(riesgo_final)

# =====================================================
# 7. INTERFAZ: RESULTADOS
# =====================================================
