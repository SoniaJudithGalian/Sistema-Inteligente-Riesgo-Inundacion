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
    # =====================================================
# TITULO
# =====================================================

st.title("🌊 Sistema Inteligente de Riesgo de Inundaciones")

st.markdown("""
Sistema híbrido basado en:

- Machine Learning
- Lógica Difusa
- API climática Open-Meteo
""")


# =====================================================
# SIDEBAR - UBICACION
# =====================================================

st.sidebar.header("🌎 Ubicación")

ciudad = st.sidebar.selectbox(
    "Seleccionar ciudad",
    [
        "Neuquén Capital",
        "Añelo",
        "Rincón de los Sauces",
        "Centenario",
        "Plottier",
        "Cutral Có"
    ]
)

coordenadas = {
    "Neuquén Capital": (-38.9516, -68.0591),
    "Añelo": (-38.3544, -68.7884),
    "Rincón de los Sauces": (-37.3906, -68.9250),
    "Centenario": (-38.8296, -68.1318),
    "Plottier": (-38.9667, -68.2333),
    "Cutral Có": (-38.9369, -69.2305)
}

latitud, longitud = coordenadas[ciudad]


# =====================================================
# DATOS CLIMATICOS
# =====================================================

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

        st.sidebar.metric(
            "🌧️ Lluvia estimada 24h",
            f"{lluvia} mm"
        )

        st.sidebar.metric(
            "💧 Humedad promedio 24h",
            f"{humedad} %"
        )

    except Exception as e:
        st.sidebar.error("No se pudieron cargar los datos climáticos")
        st.sidebar.info("Se usarán valores manuales")

        lluvia = st.sidebar.slider("🌧️ Lluvia manual (mm)", 0, 120, 50)
        humedad = st.sidebar.slider("💧 Humedad manual (%)", 0, 100, 50)
        df_clima = None

else:
    lluvia = st.sidebar.slider("🌧️ Lluvia manual (mm)", 0, 120, 50)
    humedad = st.sidebar.slider("💧 Humedad manual (%)", 0, 100, 50)
    df_clima = None


# =====================================================
# VARIABLES TERRITORIALES
# =====================================================

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


# =====================================================
# RESULTADOS
# =====================================================

st.subheader("📊 Resultados del sistema")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicción ML", prediccion_texto)

with col2:
    st.metric("Riesgo Difuso", round(riesgo_final, 2))

with col3:
    st.metric("Ciudad", ciudad)


if riesgo_final >= 80:
    st.error("🚨 RIESGO CRÍTICO DE INUNDACIÓN")
elif riesgo_final >= 60:
    st.warning("⚠️ RIESGO ALTO")
elif riesgo_final >= 40:
    st.info("🟡 RIESGO MEDIO")
else:
    st.success("🟢 RIESGO BAJO")


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
