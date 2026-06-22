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



def obtener_evento_historico_neuquen_2014():
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": -38.9516,
        "longitude": -68.0591,
        "start_date": "2014-04-02",
        "end_date": "2014-04-08",
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,wind_speed_10m_max",
        "timezone": "America/Argentina/Buenos_Aires"
    }

    respuesta = requests.get(url, params=params)
    datos = respuesta.json()

    df = pd.DataFrame(datos["daily"])

    lluvia_acumulada = df["precipitation_sum"].sum()

    return lluvia_acumulada, df



# =====================================================
# CONFIGURACION
# =====================================================

st.set_page_config(
    page_title="Riesgo de Inundaciones",
    layout="wide"
)
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

st.sidebar.markdown("##  Monitoreo territorial")

ciudades = [
    "Neuquén Capital",
    "Añelo",
    "Rincón de los Sauces",
    "Centenario",
    "Plottier",
    "Cutral Có",
    "Bahía Blanca",
    "Córdoba",
    "Buenos Aires",
    "Salta",
    "Mendoza",
    "Rosario"
]

ciudad_select = st.sidebar.selectbox(
    "Elegí una ciudad",
    ciudades
)

ciudad_manual = st.sidebar.text_input(
    "O escribí otra ciudad argentina",
    placeholder="Ejemplo: Cipolletti"
)

if "modo_bahia" not in st.session_state:
    st.session_state.modo_bahia = False

st.sidebar.markdown("---")

if st.sidebar.button(" Mostrar caso real: Bahía Blanca"):
    st.session_state.modo_bahia = True

if st.sidebar.button(" Volver a clima actual"):
    st.session_state.modo_bahia = False

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
# RESULTADOS
# =====================================================

st.markdown('<div class="section-title">📊 Resultados del sistema</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Ciudad", ciudad)

with col2:
    st.metric("Lluvia", f"{round(lluvia, 2)} mm")

with col3:
    st.metric("Predicción ML", prediccion_texto)

with col4:
    st.metric("Lógica Difusa", f"{nivel_difuso} - {round(riesgo_final, 2)}")

if riesgo_final >= 80:
    st.error("🚨 RIESGO CRÍTICO DE INUNDACIÓN")
elif riesgo_final >= 60:
    st.warning("⚠️ RIESGO ALTO")
elif riesgo_final >= 40:
    st.info("🟡 RIESGO MEDIO")
else:
    st.success("🟢 RIESGO BAJO")
# =====================================================
# MAPA TERRITORIAL
# =====================================================

st.subheader("🗺️ Visualización territorial")

def color_mapa_por_riesgo(nivel):
    nivel = str(nivel).lower()

    if "bajo" in nivel:
        return [34, 197, 94, 180]
    elif "medio" in nivel:
        return [250, 204, 21, 180]
    elif "alto" in nivel:
        return [249, 115, 22, 180]
    else:
        return [239, 68, 68, 180]


df_mapa = pd.DataFrame({
    "ciudad": [ciudad],
    "provincia": [provincia],
    "lat": [latitud],
    "lon": [longitud],
    "lluvia": [lluvia],
    "riesgo_ml": [prediccion_texto],
    "riesgo_difuso": [nivel_difuso],
    "color": [color_mapa_por_riesgo(nivel_difuso)]
})

st.pydeck_chart(
    pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(
            latitude=-38.4161,
            longitude=-63.6167,
            zoom=3.2,
            pitch=0
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df_mapa,
                get_position="[lon, lat]",
                get_radius=65000,
                get_fill_color="color",
                pickable=True
            )
        ],
        tooltip={
            "html": """
            <b>{ciudad}, {provincia}</b><br/>
            Lluvia: {lluvia} mm<br/>
            Riesgo ML: {riesgo_ml}<br/>
            Riesgo Difuso: {riesgo_difuso}
            """,
            "style": {
                "backgroundColor": "white",
                "color": "black"
            }
        }
    )
)


# =====================================================
# SEMAFORO CIRCULAR DE RIESGO DIFUSO
# =====================================================

st.subheader("🚦 Semáforo circular de Riesgo Difuso")

def crear_semaforo_circular_difuso(valor, nivel):
    fig = go.Figure()

    segmentos = [
        (0, 35, "#22C55E", "Bajo"),
        (25, 60, "#FACC15", "Medio"),
        (50, 85, "#F97316", "Alto"),
        (75, 100, "#EF4444", "Crítico")
    ]

    for inicio, fin, color, nombre in segmentos:
        theta = np.linspace(inicio * 3.6, fin * 3.6, 80)
        r = np.ones_like(theta)

        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            mode="lines",
            line=dict(color=color, width=32),
            opacity=0.72,
            name=nombre
        ))

    theta_valor = valor * 3.6

    fig.add_trace(go.Scatterpolar(
        r=[0, 0.82],
        theta=[theta_valor, theta_valor],
        mode="lines+markers",
        line=dict(color="#0F172A", width=5),
        marker=dict(size=[10, 14], color="#0F172A"),
        name="Riesgo actual"
    ))

    fig.update_layout(
        title=dict(
            text=f"Nivel actual: {nivel} | Riesgo difuso: {round(valor, 2)} / 100",
            x=0.5,
            font=dict(size=22, color="#0F172A")
        ),
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.15]),
            angularaxis=dict(
                visible=False,
                rotation=90,
                direction="clockwise"
            )
        ),
        showlegend=True,
        height=430,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=70, b=20),
        annotations=[
            dict(
                text=f"<b>{int(valor)}</b><br>{nivel.upper()}",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=28, color="#0F172A")
            )
        ]
    )

    return fig


fig_semaforo = crear_semaforo_circular_difuso(riesgo_final, nivel_difuso)

st.plotly_chart(fig_semaforo, use_container_width=True)

# =====================================================
# GRAFICO DE PERTENENCIA DEL RIESGO DIFUSO
# =====================================================

st.subheader("📉 Funciones de Pertenencia del Riesgo")

fig_riesgo = go.Figure()

fig_riesgo.add_trace(go.Scatter(
    x=riesgo.universe,
    y=riesgo["bajo"].mf,
    mode="lines",
    name="Bajo",
    line=dict(color="#22C55E", width=3)
))

fig_riesgo.add_trace(go.Scatter(
    x=riesgo.universe,
    y=riesgo["medio"].mf,
    mode="lines",
    name="Medio",
    line=dict(color="#FACC15", width=3)
))

fig_riesgo.add_trace(go.Scatter(
    x=riesgo.universe,
    y=riesgo["alto"].mf,
    mode="lines",
    name="Alto",
    line=dict(color="#EF4444", width=3)
))

fig_riesgo.add_vline(
    x=riesgo_final,
    line_width=3,
    line_dash="dash",
    line_color="#0F172A",
    annotation_text=f"Riesgo actual: {round(riesgo_final, 2)}",
    annotation_position="top"
)

fig_riesgo.update_layout(
    title="Valor del riesgo dentro de los conjuntos difusos",
    xaxis_title="Riesgo",
    yaxis_title="Grado de pertenencia",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="#0F172A"),
    height=420
)

st.plotly_chart(fig_riesgo, use_container_width=True)

# =====================================================
# GRAFICO DE VARIABLES
# =====================================================

st.subheader("📥 Variables ingresadas")

df_inputs = pd.DataFrame({
    "Variable": [
        "Lluvia",
        "Humedad",
        "Drenaje",
        "Pendiente"
    ],
    "Valor": [
        lluvia,
        humedad,
        drenaje,
        pendiente
    ]
})

fig = px.bar(
    df_inputs,
    x="Variable",
    y="Valor",
    text="Valor",
    title="Valores utilizados por el sistema"
)

st.plotly_chart(fig, use_container_width=True)


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
