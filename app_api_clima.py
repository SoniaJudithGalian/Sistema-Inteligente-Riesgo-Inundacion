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
# CONFIGURACION
# =====================================================

st.set_page_config(
    page_title="Riesgo de Inundaciones",
    layout="wide"
)
if "modo_bahia" not in st.session_state:
    st.session_state.modo_bahia = False
    
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;600;700;800&display=swap');

/* RE DUCE EL ESPACIO EN BLANCO SUPERIOR E INFERIOR DE LA APP */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

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

.card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0px 4px 18px rgba(15, 23, 42, 0.08);
}

.risk-text {
    font-size: 32px;
    font-weight: 800;
    color: #DC2626;
}
</style>
""", unsafe_allow_html=True)



# =====================================================
# CARGAR MODELO
# =====================================================

try:
    modelo = joblib.load("modelo_ml.pkl")
   
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


@st.cache_data(ttl=3600)
def buscar_ciudad_argentina(nombre_ciudad):
    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": nombre_ciudad,
        "count": 5,
        "language": "es",
        "format": "json",
        "countryCode": "AR"
    }

    try:
        respuesta = requests.get(url, params=params, timeout=10)

        if respuesta.status_code != 200:
            return []

        datos = respuesta.json()
        return datos.get("results", [])

    except Exception:
        return []


def obtener_evento_historico_bahia_blanca():
    resultados = buscar_ciudad_argentina("Bahía Blanca")

    if len(resultados) > 0:
        lugar = resultados[0]
        latitud = lugar.get("latitude")
        longitud = lugar.get("longitude")
        provincia = lugar.get("admin1", "Buenos Aires")
    else:
        latitud = -38.7196
        longitud = -62.2724
        provincia = "Buenos Aires"

    df_evento = pd.DataFrame({
        "fecha": ["07/03/2025"],
        "ciudad": ["Bahía Blanca"],
        "lluvia_mm": [290],
        "humedad_estimada": [95],
        "drenaje_estimado": [30],
        "pendiente_estimada": [2],
        "fuente": ["Argentina.gob.ar / IGN"]
    })

    return {
        "ciudad": "Bahía Blanca",
        "provincia": provincia,
        "latitud": latitud,
        "longitud": longitud,
        "lluvia": 290,
        "humedad": 95,
        "drenaje": 30,
        "pendiente": 2,
        "df_evento": df_evento
    }

    
# =====================================================
# TITULO MODERNO
# =====================================================

st.markdown(
    '<div class="main-title"> Sistema Inteligente de Riesgo de Inundaciones</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Aplicación con Machine Learning, Lógica Difusa y datos climáticos de Open-Meteo.</div>',
    unsafe_allow_html=True
)

# =====================================================
# SIDEBAR - MONITOREO
# =====================================================

st.sidebar.markdown("## 🗺️ Monitoreo territorial")

# Lista optimizada con casos críticos de estudio de inundaciones en Argentina
ciudades = [
    # Región Comahue / Localidades del Alto Valle y alrededores
    "Neuquén Capital",
    "Plottier",
    "Centenario",
    "Añelo",
    "Rincón de los Sauces",
    "Cutral Có",
    "Cipolletti",
    "General Roca",
    
    # Casos emblemáticos e históricos de inundaciones en el país
    "La Plata",             # Crítico por inundaciones pluviales urbanas
    "Santa Fe Capital",     # Histórico por crecidas del Río Salado / Paraná
    "Resistencia",          # Vulnerable por sistemas fluviales y precipitaciones intensas
    "Comodoro Rivadavia",   # Crítico por escorrentía e inundaciones extraordinarias
    "Concordia",            # Afectado frecuentemente por las crecidas del Río Uruguay
    
    # Grandes centros urbanos y otras regiones
    "Buenos Aires",
    "Córdoba",
    "Rosario",
    "Salta",
    "Mendoza"
]

ciudad_select = st.sidebar.selectbox(
    "Elegí una ciudad para analizar",
    ciudades
)

ciudad_manual = st.sidebar.text_input(
    "O escribí otra ciudad argentina",
    placeholder="Ejemplo: Viedma"
)



# =====================================================
# DATOS SEGÚN MODO
# =====================================================

df_clima = None
df_evento_bahia = None

if st.session_state.modo_bahia:

    evento = obtener_evento_historico_bahia_blanca()

    ciudad = evento["ciudad"]
    provincia = evento["provincia"]
    latitud = evento["latitud"]
    longitud = evento["longitud"]

    lluvia = evento["lluvia"]
    humedad = evento["humedad"]
    drenaje = evento["drenaje"]
    pendiente = evento["pendiente"]

    df_evento_bahia = evento["df_evento"]

    st.sidebar.info("Caso histórico cargado: Bahía Blanca 2025")

else:

    ciudad_a_buscar = ciudad_manual if ciudad_manual.strip() != "" else ciudad_select

    resultados = buscar_ciudad_argentina(ciudad_a_buscar)

    if len(resultados) == 0:
        st.error("No se encontró la ciudad. Probá escribir el nombre de otra forma.")
        st.stop()

    lugar = resultados[0]

    ciudad = lugar.get("name", ciudad_a_buscar)
    provincia = lugar.get("admin1", "")
    latitud = lugar.get("latitude")
    longitud = lugar.get("longitude")

    st.sidebar.header("🌦️ Datos climáticos")

    usar_api = st.sidebar.checkbox(
        "Usar datos climáticos automáticos",
        value=True
    )

    if usar_api:
        try:
            lluvia_api, humedad_api, df_clima = obtener_clima_open_meteo(
                latitud,
                longitud
            )

            lluvia = round(float(lluvia_api), 2)
            humedad = round(float(humedad_api), 2)

            st.sidebar.success("Datos climáticos cargados")

            st.sidebar.metric("🌧️ Lluvia estimada 24h", f"{lluvia} mm")
            st.sidebar.metric("💧 Humedad promedio 24h", f"{humedad} %")

        except Exception as e:
            st.sidebar.error("No se pudieron cargar los datos climáticos")
            lluvia = st.sidebar.slider("🌧️ Lluvia manual (mm)", 0, 300, 50)
            humedad = st.sidebar.slider("💧 Humedad manual (%)", 0, 100, 50)
            df_clima = None

    else:
        lluvia = st.sidebar.slider("🌧️ Lluvia manual (mm)", 0, 300, 50)
        humedad = st.sidebar.slider("💧 Humedad manual (%)", 0, 100, 50)
        df_clima = None

    st.sidebar.header("🏙️ Variables territoriales")

    drenaje = st.sidebar.slider(
        "🚰 Capacidad de drenaje (%)",
        0,
        100,
        50
    )

    pendiente = st.sidebar.slider(
        "⛰️ Pendiente topográfica (%)",
        0,
        10,
        5
    )



# =====================================================
# MOSTRAR DATOS DEL CASO BAHÍA BLANCA
# =====================================================

if st.session_state.modo_bahia:
    st.markdown('<div class="section-title">📌 Caso real: Bahía Blanca</div>', unsafe_allow_html=True)

    st.warning(
        "Se cargó el evento histórico de Bahía Blanca del 07/03/2025, "
        "con aproximadamente 290 mm de lluvia en 12 horas."
    )

    st.dataframe(df_evento_bahia, use_container_width=True)


# =====================================================
# MACHINE LEARNING
# =====================================================

nuevo_caso = pd.DataFrame({
    "lluvia_mm": [lluvia],
    "humedad_suelo": [humedad],
    "capacidad_drenaje": [drenaje],
    "pendiente_topografica": [pendiente]
})

prediccion_ml = modelo.predict(nuevo_caso)

clases_riesgo = {
    0: "🟢 Bajo",
    1: "🟡 Medio",
    2: "🟠 Alto",
    3: "🔴 Crítico"
}

prediccion_texto = clases_riesgo.get(
    int(prediccion_ml[0]),
    "Desconocido"
)


# =====================================================
# LOGICA DIFUSA
# =====================================================

lluvia_fz = ctrl.Antecedent(np.arange(0, 121, 1), "lluvia")
humedad_fz = ctrl.Antecedent(np.arange(0, 101, 1), "humedad")
drenaje_fz = ctrl.Antecedent(np.arange(0, 101, 1), "drenaje")
pendiente_fz = ctrl.Antecedent(np.arange(0, 11, 1), "pendiente")

riesgo = ctrl.Consequent(np.arange(0, 101, 1), "riesgo")

lluvia_fz["baja"] = fuzz.trimf(lluvia_fz.universe, [0, 0, 40])
lluvia_fz["media"] = fuzz.trimf(lluvia_fz.universe, [20, 60, 90])
lluvia_fz["alta"] = fuzz.trimf(lluvia_fz.universe, [70, 120, 120])

humedad_fz["seca"] = fuzz.trimf(humedad_fz.universe, [0, 0, 40])
humedad_fz["media"] = fuzz.trimf(humedad_fz.universe, [20, 50, 80])
humedad_fz["saturada"] = fuzz.trimf(humedad_fz.universe, [60, 100, 100])

drenaje_fz["bajo"] = fuzz.trimf(drenaje_fz.universe, [0, 0, 40])
drenaje_fz["medio"] = fuzz.trimf(drenaje_fz.universe, [20, 50, 80])
drenaje_fz["alto"] = fuzz.trimf(drenaje_fz.universe, [60, 100, 100])

pendiente_fz["llana"] = fuzz.trimf(pendiente_fz.universe, [0, 0, 3])
pendiente_fz["moderada"] = fuzz.trimf(pendiente_fz.universe, [2, 5, 8])
pendiente_fz["empinada"] = fuzz.trimf(pendiente_fz.universe, [6, 10, 10])

riesgo["bajo"] = fuzz.trimf(riesgo.universe, [0, 0, 40])
riesgo["medio"] = fuzz.trimf(riesgo.universe, [20, 50, 80])
riesgo["alto"] = fuzz.trimf(riesgo.universe, [60, 100, 100])


# =====================================================
# REGLAS DIFUSAS
# =====================================================

regla1 = ctrl.Rule(
    lluvia_fz["alta"] & humedad_fz["saturada"] & drenaje_fz["bajo"],
    riesgo["alto"]
)

regla2 = ctrl.Rule(
    lluvia_fz["media"] & drenaje_fz["medio"],
    riesgo["medio"]
)

regla3 = ctrl.Rule(
    lluvia_fz["baja"] & drenaje_fz["alto"],
    riesgo["bajo"]
)

regla4 = ctrl.Rule(
    pendiente_fz["llana"] & lluvia_fz["alta"],
    riesgo["alto"]
)

regla5 = ctrl.Rule(
    lluvia_fz["alta"],
    riesgo["alto"]
)

regla6 = ctrl.Rule(
    lluvia_fz["media"],
    riesgo["medio"]
)

regla7 = ctrl.Rule(
    lluvia_fz["baja"],
    riesgo["bajo"]
)

regla8 = ctrl.Rule(
    (lluvia_fz["media"] | lluvia_fz["alta"]) & drenaje_fz["bajo"],
    riesgo["alto"]
)

regla9 = ctrl.Rule(
    drenaje_fz["alto"],
    riesgo["bajo"]
)


sistema = ctrl.ControlSystem([
    regla1,
    regla2,
    regla3,
    regla4,
    regla5,
    regla6,
    regla7,
    regla8,
    regla9
])

simulador = ctrl.ControlSystemSimulation(sistema)

simulador.input["lluvia"] = lluvia
simulador.input["humedad"] = humedad
simulador.input["drenaje"] = drenaje
simulador.input["pendiente"] = pendiente

simulador.compute()

riesgo_final = simulador.output.get("riesgo", 0)

def obtener_nivel_difuso(valor):
    if valor >= 80:
        return "Crítico", "#EF4444"
    elif valor >= 60:
        return "Alto", "#F97316"
    elif valor >= 40:
        return "Medio", "#FACC15"
    else:
        return "Bajo", "#22C55E"


nivel_difuso, color_activo = obtener_nivel_difuso(riesgo_final)

# =====================================================
# RESULTADOS (KPIs)
# =====================================================
st.markdown('<div class="section-title">📊 Resumen de Situación</div>', unsafe_allow_html=True)

# Usamos un contenedor nativo de Streamlit para darle un marco profesional
with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📍 Ciudad", ciudad)
    with col2:
        st.metric("🌧️ Lluvia", f"{round(lluvia, 2)} mm")
    with col3:
        st.metric("🤖 Predicción ML", prediccion_texto)
    with col4:
        st.metric("🧠 Lógica Difusa", f"{nivel_difuso} ({round(riesgo_final, 1)})")

# Mantenemos las alertas, pero sin tanto espacio
if riesgo_final >= 80:
    st.error("🚨 RIESGO CRÍTICO DE INUNDACIÓN")
elif riesgo_final >= 60:
    st.warning("⚠️ RIESGO ALTO")
elif riesgo_final >= 40:
    st.info("🟡 RIESGO MEDIO")
else:
    st.success("🟢 RIESGO BAJO")



# =====================================================
# MAPA TERRITORIAL 3D Y SEMÁFORO (LADO A LADO)
# =====================================================
st.markdown('<div class="section-title"> Análisis Territorial y Nivel de Riesgo</div>', unsafe_allow_html=True)

# Creamos las columnas: el mapa ocupa más espacio (1.5) que el semáforo (1)
col_mapa, col_semaforo = st.columns([1.5, 1])

# ----------------- COLUMNA IZQUIERDA: EL MAPA 3D -----------------
with col_mapa:
    
    def elevacion_por_riesgo(riesgo_num):
        return max(1000, riesgo_num * 50) 
        
    df_mapa = pd.DataFrame({
        "ciudad": [ciudad],
        "provincia": [provincia],
        "lat": [latitud],
        "lon": [longitud],
        "lluvia": [lluvia],
        "riesgo_ml": [prediccion_texto],
        "riesgo_difuso": [nivel_difuso],
        "color": [color_mapa_por_riesgo(nivel_difuso)],
        "altura_cilindro": [elevacion_por_riesgo(riesgo_final)]
    })

    capa_terreno = pdk.Layer(
        "TerrainLayer",
        elevation_decoder={"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256, "offset": -32768},
        texture="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        elevation_data="https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
    )

    capa_riesgo = pdk.Layer(
        "ColumnLayer",
        data=df_mapa,
        get_position="[lon, lat]",
        get_elevation="altura_cilindro",
        get_fill_color="color",
        radius=2000,
        elevation_scale=50,
        pickable=True,
        extruded=True,
    )

    vista_inicial = pdk.ViewState(
        latitude=latitud,
        longitude=longitud,
        zoom=10.5,
        pitch=65,
        bearing=15
    )

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=vista_inicial,
            layers=[capa_terreno, capa_riesgo],
            tooltip={"html": "<b>{ciudad}</b><br/>Riesgo: {riesgo_difuso}"} # Tooltip simplificado para el ejemplo
        ),
        use_container_width=True
    )

# ----------------- COLUMNA DERECHA: EL SEMÁFORO -----------------
with col_semaforo:
    
    # Llamás a la función original que ya tenías escrita en tu código
    fig_semaforo = crear_semaforo_circular_difuso(riesgo_final, nivel_difuso)
    
    # Le ajustamos los márgenes para que no quede con tanto espacio blanco
    fig_semaforo.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
    
    st.plotly_chart(fig_semaforo, use_container_width=True)

# =====================================================
# GRÁFICOS SECUNDARIOS (DOS COLUMNAS)
# =====================================================
st.markdown('<div class="section-title">📈 Desglose de Datos y Pronóstico</div>', unsafe_allow_html=True)

col_graficos_izq, col_graficos_der = st.columns(2)

with col_graficos_izq:
    st.markdown("**Funciones de Pertenencia del Riesgo**")
    # (Pegá acá tu código de fig_riesgo)
    fig_riesgo.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=300)
    st.plotly_chart(fig_riesgo, use_container_width=True)

    st.markdown("**Variables Utilizadas**")
    # (Pegá acá tu código de px.bar para df_inputs)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=250)
    st.plotly_chart(fig, use_container_width=True)

with col_graficos_der:
    if df_clima is not None:
        st.markdown(f"**🌦️ Pronóstico para {ciudad}**")
        
        # Gráfico de Lluvia
        fig_lluvia.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=250)
        st.plotly_chart(fig_lluvia, use_container_width=True)
        
        # Gráfico de Humedad
        fig_humedad.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=250)
        st.plotly_chart(fig_humedad, use_container_width=True)
    else:
        st.info("Ingreso de datos manual activado. No hay pronóstico horario disponible.")

# =====================================================
# GRAFICO CLIMATICO API
# =====================================================

if df_clima is not None:

    st.subheader(f"🌦️ Pronóstico climático para {ciudad}")

    df_clima["time"] = pd.to_datetime(df_clima["time"])

    fig_lluvia = px.bar(
        df_clima,
        x="time",
        y="precipitation",
        title="Precipitación estimada por hora",
        labels={
            "time": "Hora",
            "precipitation": "Precipitación (mm)"
        }
    )

    st.plotly_chart(fig_lluvia, use_container_width=True)

    fig_humedad = px.line(
        df_clima,
        x="time",
        y="relative_humidity_2m",
        title="Humedad relativa por hora",
        labels={
            "time": "Hora",
            "relative_humidity_2m": "Humedad (%)"
        }
    )

    st.plotly_chart(fig_humedad, use_container_width=True)


# =====================================================
# INTERPRETACION
# =====================================================

st.subheader("🧠 Interpretación")

st.write(f"""
El sistema analiza el riesgo de inundación para **{ciudad}**.

La lluvia y la humedad pueden obtenerse automáticamente desde una API climática.
La capacidad de drenaje y la pendiente topográfica se ingresan manualmente porque dependen
de las características territoriales de cada zona.

El modelo de Machine Learning clasifica el escenario, mientras que la lógica difusa permite
interpretar el riesgo de forma gradual.
""")
