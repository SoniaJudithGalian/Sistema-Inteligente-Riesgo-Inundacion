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

/* Reduce el espacio en blanco de la parte superior */
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title { font-size: 46px; font-weight: 800; color: #0F172A; margin-bottom: 0px; }
.subtitle { font-size: 18px; color: #475569; margin-bottom: 25px; }
.section-title { font-size: 28px; font-weight: 700; color: #0F766E; margin-top: 25px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. CARGAR MODELO ML
# =====================================================

try:
    modelo = joblib.load("modelo_ml.pkl")
except Exception as e:
    st.error("Error al cargar el modelo ML. Verifica el archivo 'modelo_ml.pkl'")
    st.stop()

# =====================================================
# 3. FUNCIONES DE CLIMA Y GEOLOCALIZACIÓN
# =====================================================

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

@st.cache_data(ttl=3600)
def buscar_ciudad_argentina(nombre_ciudad):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": nombre_ciudad, "count": 1, "language": "es", "format": "json", "countryCode": "AR"}
    try:
        respuesta = requests.get(url, params=params, timeout=10)
        if respuesta.status_code == 200 and "results" in respuesta.json():
            return respuesta.json()["results"][0]
        return None
    except Exception:
        return None

# =====================================================
# 4. TÍTULO PRINCIPAL
# =====================================================

st.markdown('<div class="main-title">Sistema Inteligente de Riesgo de Inundaciones</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Aplicación con Machine Learning, Lógica Difusa y datos climáticos de Open-Meteo.</div>', unsafe_allow_html=True)

# =====================================================
# 5. SIDEBAR Y SELECCIÓN DE DATOS
# =====================================================

st.sidebar.markdown("## Monitoreo territorial")

ciudades = [
    "Neuquén Capital", "Plottier", "Centenario", "Añelo", 
    "Cipolletti", "General Roca", "La Plata", "Santa Fe Capital", 
    "Resistencia", "Comodoro Rivadavia", "Concordia", "Buenos Aires", 
    "Córdoba", "Rosario", "Salta", "Mendoza", 
    "Bahía Blanca (Caso Histórico 2025)"
]

ciudad_select = st.sidebar.selectbox("Elegí una ciudad para analizar", ciudades)

df_clima = None
es_caso_historico = False

if ciudad_select == "Bahía Blanca (Caso Histórico 2025)":
    es_caso_historico = True
    ciudad = "Bahía Blanca"
    provincia = "Buenos Aires"
    latitud, longitud = -38.7196, -62.2724
    lluvia = 290.0
    humedad = 95.0
    drenaje = 30
    pendiente = 2
    
    st.sidebar.warning("⚠️ Modo Simulación: Evento histórico extremo cargado.")

else:
    lugar = buscar_ciudad_argentina(ciudad_select)
    if lugar is None:
        st.error("No se encontraron coordenadas para la ciudad.")
        st.stop()

    ciudad = lugar.get("name", ciudad_select)
    provincia = lugar.get("admin1", "")
    latitud = lugar.get("latitude")
    longitud = lugar.get("longitude")

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

st.markdown('<div class="section-title">Resumen de Situación</div>', unsafe_allow_html=True)

with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Ciudad", ciudad)
    with col2: st.metric("Lluvia Total", f"{round(lluvia, 1)} mm")
    with col3: st.metric("Modelo ML", prediccion_texto)
    with col4: st.metric("Mod. Difuso", f"{nivel_difuso} ({int(riesgo_final)})")

if riesgo_final >= 80: st.error("RIESGO CRÍTICO DE INUNDACIÓN")
elif riesgo_final >= 60: st.warning("RIESGO ALTO")
elif riesgo_final >= 40: st.info("RIESGO MEDIO")
else: st.success("RIESGO BAJO")

# =====================================================
# 8. INTERFAZ: MAPA 3D Y SEMÁFORO
# =====================================================

st.markdown('<div class="section-title">Análisis Territorial</div>', unsafe_allow_html=True)

col_mapa, col_semaforo = st.columns([1.5, 1])

with col_mapa:
    def color_mapa(nivel):
        if nivel == "Bajo": return [34, 197, 94, 200]
        elif nivel == "Medio": return [250, 204, 21, 200]
        elif nivel == "Alto": return [249, 115, 22, 200]
        else: return [239, 68, 68, 200]

    df_mapa = pd.DataFrame({
        "lat": [latitud], "lon": [longitud], 
        "color": [color_mapa(nivel_difuso)], 
        "altura": [max(1000, riesgo_final * 80)]
    })

    vista = pdk.ViewState(latitude=latitud, longitude=longitud, zoom=10, pitch=50)
    
    capa_riesgo = pdk.Layer(
        "ColumnLayer", data=df_mapa, get_position="[lon, lat]", get_elevation="altura",
        get_fill_color="color", radius=2500, elevation_scale=50, pickable=True, extruded=True,
    )

    st.pydeck_chart(pdk.Deck(
        map_style="light",
        initial_view_state=vista, layers=[capa_riesgo],
        tooltip={"html": f"<b>{ciudad}</b><br/>Riesgo Difuso: {nivel_difuso}"}
    ), use_container_width=True)

with col_semaforo:
    def crear_semaforo(valor, nivel):
        fig = go.Figure()
        segmentos = [(0, 35, "#22C55E"), (25, 60, "#FACC15"), (50, 85, "#F97316"), (75, 100, "#EF4444")]
        for inicio, fin, color in segmentos:
            theta = np.linspace(inicio * 3.6, fin * 3.6, 80)
            fig.add_trace(go.Scatterpolar(r=np.ones_like(theta), theta=theta, mode="lines", line=dict(color=color, width=32), opacity=0.8, showlegend=False))
        
        theta_valor = valor * 3.6
        fig.add_trace(go.Scatterpolar(r=[0, 0.85], theta=[theta_valor, theta_valor], mode="lines+markers", line=dict(color="#0F172A", width=5), marker=dict(size=[1, 14], color="#0F172A"), showlegend=False))
        
        fig.update_layout(
            title=dict(text=f"Riesgo: {round(valor, 1)} / 100", x=0.5, font=dict(size=18)),
            polar=dict(radialaxis=dict(visible=False, range=[0, 1.1]), angularaxis=dict(visible=False, rotation=90, direction="clockwise")),
            height=380, margin=dict(l=10, r=10, t=40, b=10),
            annotations=[dict(text=f"<b>{int(valor)}</b><br>{nivel.upper()}", x=0.5, y=0.5, showarrow=False, font=dict(size=22))]
        )
        return fig

    st.plotly_chart(crear_semaforo(riesgo_final, nivel_difuso), use_container_width=True)

# =====================================================
# 9. INTERFAZ: GRÁFICOS SECUNDARIOS
# =====================================================

st.markdown('<div class="section-title">Desglose de Datos</div>', unsafe_allow_html=True)
col_graficos_izq, col_graficos_der = st.columns(2)

with col_graficos_izq:
    st.markdown("**Variables Utilizadas**")
    df_inputs = pd.DataFrame({"Variable": ["Lluvia", "Humedad", "Drenaje", "Pendiente"], "Valor": [lluvia, humedad, drenaje, pendiente]})
    fig_bar = px.bar(df_inputs, x="Variable", y="Valor", text="Valor", color_discrete_sequence=["#0F766E"])
    fig_bar.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)

with col_graficos_der:
    if es_caso_historico:
        st.info("Visualizando caso histórico. No hay pronóstico horario futuro para esta fecha.")
    elif df_clima is not None:
        st.markdown(f"**Pronóstico horario ({ciudad})**")
        df_clima["time"] = pd.to_datetime(df_clima["time"])
        fig_lluvia = px.bar(df_clima, x="time", y="precipitation", labels={"time": "", "precipitation": "mm"}, color_discrete_sequence=["#3B82F6"])
        fig_lluvia.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_lluvia, use_container_width=True)
    else:
        st.info("Ingreso manual activado. No hay pronóstico horario de la API.")
