import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import datetime
import requests
import folium
from streamlit_folium import st_folium

# =====================================================
# 1. CONFIGURACION DE LA PÁGINA Y ESTILOS (Limpios)
# =====================================================

st.set_page_config(
    page_title="Riesgo de Inundaciones",
    layout="wide"
)

# Eliminamos emojis de la barra de título principal
st.title("Riesgo de Inundaciones")

st.markdown("""
Aplicación con Machine Learning, Lógica Difusa y datos climáticos de Open-Meteo.
""")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-size: 46px;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 18px;
    color: #475569;
    margin-bottom: 25px;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    color: #0F766E;
    margin-top: 25px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. BASE DE DATOS DE COORDENADAS (Unificada)
# =====================================================

ciudades_coordenadas = {
    "Bahía Blanca": {"lat": -38.7196, "lon": -62.2724, "provincia": "Buenos Aires"},
    "Cipolletti": {"lat": -38.9406, "lon": -67.9902, "provincia": "Río Negro"},
    "Neuquén Capital": {"lat": -38.9516, "lon": -68.0591, "provincia": "Neuquén"}, # Agregada
    "Plottier": {"lat": -38.9672, "lon": -68.2292, "provincia": "Neuquén"}
}

# =====================================================
# 3. FUNCIONES DE APOYO Y APIS
# =====================================================

@st.cache_data
def load_historical_data():
    try:
        data = pd.read_csv("historicos.csv")
        return data
    except FileNotFoundError:
        st.error("No se encontró el archivo 'historicos.csv'.")
        return pd.DataFrame()

# Cargar el modelo pre-entrenado
try:
    modelo_ml = joblib.load("modelo_lluvia.pkl")
except FileNotFoundError:
    st.error("No se encontró el archivo 'modelo_lluvia.pkl'.")
    modelo_ml = None

# =====================================================
# 4. CONFIGURACIÓN DEL MAPA CON UBICACIÓN (Nuevo)
# =====================================================

# Modificamos la función para agregar el marcador circular limpio
def create_map(lat, lon, city_name, zoom=10):
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles='cartodbpositron')
    
    # Agregamos el marcador circular (profesional y limpio)
    folium.CircleMarker(
        location=[lat, lon],
        radius=8,  # Tamaño del punto
        color='#0F766E',  # Color del tema de la app
        fill=True,
        fill_color='#0F766E',
        fill_opacity=0.7,
        tooltip=f"Ubicación: {city_name}" # Muestra nombre al pasar mouse
    ).add_to(m)
    
    return m

# =====================================================
# 5. SIDEBAR (Limpios y Ampliados)
# =====================================================

# Eliminamos emoji de la barra lateral
st.sidebar.title("Monitoreo territorial")

ciudad_select = st.sidebar.selectbox(
    "Elegí una ciudad para analizar",
    list(ciudades_coordenadas.keys())
)

# Variables territoriales (estáticas por ahora)
if ciudad_select in ciudades_coordenadas:
    territorial = pd.DataFrame({
        "Variable": ["Suelo Impermeable", "Red Pluvial", "Pendiente"],
        "Valor": [75.0, 60.0, 1.2]
    })
    
    lat = ciudades_coordenadas[ciudad_select]["lat"]
    lon = ciudades_coordenadas[ciudad_select]["lon"]
    provincia = ciudades_coordenadas[ciudad_select]["provincia"]
else:
    territorial = pd.DataFrame()
    st.sidebar.error("Datos territoriales no disponibles.")

# =====================================================
# 6. MONITOREO CLIMÁTICO
# =====================================================

lluvia_api = 0.0
humedad_api = 0.0
clima_desc = "Cargando..."

if st.sidebar.button("Actualizar clima"):
    if ciudad_select in ciudades_coordenadas:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation,relative_humidity_2m&forecast_days=1&timezone=auto"
        
        try:
            response = requests.get(url)
            clima_data = response.json()
            lluvia_api = clima_data["hourly"]["precipitation"][0]
            humedad_api = clima_data["hourly"]["relative_humidity_2m"][0]
            clima_desc = "Actualizado"
        except requests.exceptions.RequestException:
            st.error("Error al conectar con la API de clima.")

st.sidebar.markdown(f"**Ubicación:** {provincia}")
st.sidebar.markdown(f"**Estado clima:** {clima_desc}")

if "clima_manual" not in st.session_state:
    st.session_state.clima_manual = False

if st.sidebar.button("Ingresar datos manuales"):
    st.session_state.clima_manual = not st.session_state.clima_manual

if st.session_state.clima_manual:
    lluvia_manual = st.sidebar.slider("Lluvia (mm)", 0.0, 300.0, 50.0)
    humedad_manual = st.sidebar.slider("Humedad (%)", 0, 100, 70)
else:
    lluvia_manual = 0.0
    humedad_manual = 0.0

# Definir las variables de clima finales
lluvia = lluvia_api if not st.session_state.clima_manual else lluvia_manual
humedad = humedad_api if not st.session_state.clima_manual else humedad_manual

# =====================================================
# 7. PROCESAMIENTO CON ML Y LÓGICA DIFUSA
# =====================================================

# ML: Predicción de lluvia (modelo simulado)
# Usando joblib cargado al principio (lluvia -> lluvia (modelo_ml))
# Nota: Este código asume que el modelo predice *si va a llover* (0/1) 
# o la *cantidad*, pero aquí lo usamos para calcular el riesgo.
ml_pred = 0 # Valor por defecto
if modelo_ml:
    # Preparar el input para el modelo simulado (Lluvia vs Lluvia ML)
    # Supongamos que el modelo necesita una matriz 1x1
    input_model = np.array([[lluvia]]) 
    ml_pred = modelo_ml.predict(input_model)[0]

# Lógica Difusa: Predicción de riesgo (simulada)
risk_percentage = 0
fuzzy_risk = "Nivel Difuso"

# Usamos valores para simular riesgo difuso entre 0 y 100
# Basado en la lluvia manual o de la API
if lluvia > 150:
    risk_percentage = 95
elif lluvia > 75:
    risk_percentage = 65
elif lluvia > 25:
    risk_percentage = 35
else:
    risk_percentage = 15

# Definimos el texto según el porcentaje simulado
if risk_percentage >= 80:
    fuzzy_risk = "Crítico"
elif risk_percentage >= 50:
    fuzzy_risk = "Alto"
elif risk_percentage >= 25:
    fuzzy_risk = "Medio"
else:
    fuzzy_risk = "Bajo"

# Determinar prediccion_texto para ML basado en el valor predicho (0 o 1)
prediccion_texto = "Probable" if ml_pred == 1 else "Baja"
nivel_difuso = fuzzy_risk # Mantenemos el nombre de la variable para compatibilidad

# =====================================================
# 8. CASO HISTÓRICO: BAHÍA BLANCA 2025 (Limpios)
# =====================================================

if st.sidebar.button("Cargar caso histórico Bahía Blanca"):
    if "clima_historico" not in st.session_state:
        st.session_state.clima_historico = {}
        st.session_state.case_fuzzy_risk = {}
        st.session_state.case_risk_percentage = {}
        
    st.session_state.clima_historico["case1"] = pd.DataFrame({
        "Lluvia": [290.0, 5.0, 2.0],
        "Suelo Impermeable": [75.0, 75.0, 75.0],
        "capacidad_drenaje": [60.0, 60.0, 60.0],
        "Fecha": ["17-12-2023", "18-12-2023", "19-12-2023"]
    })
    # Simular riesgo para el caso histórico
    st.session_state.case_fuzzy_risk["case1"] = "Crítico"
    st.session_state.case_risk_percentage["case1"] = 98

# Limpiar caso histórico
if st.sidebar.button("Limpiar monitoreo territorial"):
    if "clima_historico" in st.session_state:
        del st.session_state.clima_historico
        del st.session_state.case_fuzzy_risk
        del st.session_state.case_risk_percentage
    st.success("Monitoreo territorial limpiado.")

# =====================================================
# 9. INTERFAZ: RESULTADOS (Limpios)
# =====================================================

# Título de sección sin emoji
st.markdown('<div class="section-title">Resumen de Situación</div>', unsafe_allow_html=True)

# Tarjetas de métricas sin emojis
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1: 
        st.metric("Ciudad Seleccionada", ciudad_select)
        st.metric("Modelo de Machine Learning", prediccion_texto)
    with col2: 
        st.metric("Lluvia Registrada", f"{round(lluvia, 1)} mm")
        st.metric("Sistema Difuso", f"{nivel_difuso} ({int(risk_percentage)} / 100)")

# Alertas sin emojis
if risk_percentage >= 80: 
    st.error("RIESGO CRÍTICO DE INUNDACIÓN")
elif risk_percentage >= 50: 
    st.warning("RIESGO ALTO")
elif risk_percentage >= 25: 
    st.info("RIESGO MEDIO")
else: 
    st.success("RIESGO BAJO")

# =====================================================
# 10. INTERFAZ: MAPA Y SEMÁFORO CON UBICACIÓN (Nuevo)
# =====================================================

# Título de sección sin emoji
st.markdown('<div class="section-title">Análisis Territorial</div>', unsafe_allow_html=True)

col_mapa, col_semaforo = st.columns([1.5, 1])

with col_mapa:
    if ciudad_select in ciudades_coordenadas:
        # Definimos el zoom según el tamaño de la ciudad
        zoom_level = 12 if ciudad_select in ["Plottier", "Cipolletti"] else 11
        
        # Llamamos a la función create_map pasando el nombre de la ciudad
        m = create_map(lat, lon, ciudad_select, zoom=zoom_level)
        
        # Renderizamos el mapa con st_folium
        st_folium(m, width=700, height=380, returned_objects=[])
    else:
        st.error("Error al cargar las coordenadas para el mapa.")

# =====================================================
# 11. INTERFAZ: GRÁFICOS Y CASOS HISTÓRICOS
# =====================================================

# Mostrar caso histórico si está cargado
if "clima_historico" in st.session_state and "case1" in st.session_state.clima_historico:
    st.markdown('<div class="section-title">Caso Histórico: Temporal de Lluvia Bahía Blanca - dic 2023</div>', unsafe_allow_html=True)
    
    st.markdown(f"**Ubicación:** Bahía Blanca, Buenos Aires. **Fecha del evento:** 17-12-2023 al 19-12-2023")
    
    # Simulación de riesgo del caso histórico (usando st_folium NO es dinámico, 
    # pero aquí mostramos la tabla)
    case_risk = st.session_state.case_fuzzy_risk["case1"]
    case_percentage = st.session_state.case_risk_percentage["case1"]
    
    # Tarjeta de métricas para el caso histórico (Limpios)
    with st.container(border=True):
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1: st.metric("Modelo de Machine Learning", "Probable")
        with col_c2: st.metric("Lluvia Registrada", "290 mm")
        with col_c3: st.metric("Sistema Difuso", f"{case_risk} ({int(case_percentage)} / 100)")
        
    st.error(f"Alerta: Riesgo {case_risk} simulado para la fecha del evento.")
    st.write(st.session_state.clima_historico["case1"])
else:
    # st.markdown('<div class="section-title">Datos territoriales</div>', unsafe_allow_html=True)
    st.write("") # Espacio para no mostrar el titulo si no hay caso historico

# Gráfico de predicción de lluvia (ML vs Manual)
st.markdown('<div class="section-title">Predicción de Lluvia (Machine Learning)</div>', unsafe_allow_html=True)

if modelo_ml:
    # Usando joblib cargado al principio (lluvia -> lluvia (modelo_ml))
    input_model = np.array([[lluvia]]) 
    ml_pred = modelo_ml.predict(input_model)[0]

    st.markdown(f"**Lluvia simulada por el modelo ML:** {ml_pred} mm.")
    
    col1, col2 = st.columns([1,1])
    
    # Gráfico de dispersión para comparar lluvia vs Lluvia ML
    df_pred = pd.DataFrame({'Variable':['Actual (mm)', 'Modelo ML (mm)'], 'Lluvia':[lluvia, ml_pred]})
    
    with col1:
        fig_pred = px.bar(df_pred, x='Variable', y='Lluvia', title='Comparación de Lluvia vs Predicción de Modelo ML')
        st.plotly_chart(fig_pred)

    with col2:
        st.success(f"La predicción de lluvia del modelo es {prediccion_texto}.")

# Título de sección de datos territoriales
st.markdown('<div class="section-title">Datos Territoriales</div>', unsafe_allow_html=True)
st.write(territorial)
