import streamlit as st
import pandas as pd
import numpy as np
import joblib

import skfuzzy as fuzz
from skfuzzy import control as ctrl

# =====================================================
# CONFIGURACION
# =====================================================

st.set_page_config(
    page_title="Riesgo de Inundaciones",
    layout="wide"
)

# =====================================================
# CARGAR MODELO
# =====================================================

modelo = joblib.load("modelo_ml.pkl")

# =====================================================
# TITULO
# =====================================================

st.title("🌊 Sistema Inteligente de Riesgo de Inundaciones")

st.markdown("""
Sistema híbrido basado en:

- Machine Learning
- Lógica Difusa
- Inteligencia Artificial
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Variables de Entrada")

lluvia = st.sidebar.slider(
    "🌧️ Lluvia (mm)",
    0,
    120,
    50
)

humedad = st.sidebar.slider(
    "💧 Humedad del suelo (%)",
    0,
    100,
    50
)

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

# =====================================================
# LOGICA DIFUSA
# =====================================================

lluvia_fz = ctrl.Antecedent(
    np.arange(0,121,1),
    'lluvia'
)

humedad_fz = ctrl.Antecedent(
    np.arange(0,101,1),
    'humedad'
)

drenaje_fz = ctrl.Antecedent(
    np.arange(0,101,1),
    'drenaje'
)

pendiente_fz = ctrl.Antecedent(
    np.arange(0,11,1),
    'pendiente'
)

riesgo = ctrl.Consequent(
    np.arange(0,101,1),
    'riesgo'
)

# =====================================================
# FUNCIONES DE MEMBRESIA
# =====================================================

lluvia_fz['baja'] = fuzz.trimf(
    lluvia_fz.universe,
    [0,0,40]
)

lluvia_fz['media'] = fuzz.trimf(
    lluvia_fz.universe,
    [20,60,90]
)

lluvia_fz['alta'] = fuzz.trimf(
    lluvia_fz.universe,
    [70,120,120]
)

humedad_fz['seca'] = fuzz.trimf(
    humedad_fz.universe,
    [0,0,40]
)

humedad_fz['media'] = fuzz.trimf(
    humedad_fz.universe,
    [20,50,80]
)

humedad_fz['saturada'] = fuzz.trimf(
    humedad_fz.universe,
    [60,100,100]
)

drenaje_fz['bajo'] = fuzz.trimf(
    drenaje_fz.universe,
    [0,0,40]
)

drenaje_fz['medio'] = fuzz.trimf(
    drenaje_fz.universe,
    [20,50,80]
)

drenaje_fz['alto'] = fuzz.trimf(
    drenaje_fz.universe,
    [60,100,100]
)

pendiente_fz['llana'] = fuzz.trimf(
    pendiente_fz.universe,
    [0,0,3]
)

pendiente_fz['moderada'] = fuzz.trimf(
    pendiente_fz.universe,
    [2,5,8]
)

pendiente_fz['empinada'] = fuzz.trimf(
    pendiente_fz.universe,
    [6,10,10]
)

riesgo['bajo'] = fuzz.trimf(
    riesgo.universe,
    [0,0,40]
)

riesgo['medio'] = fuzz.trimf(
    riesgo.universe,
    [20,50,80]
)

riesgo['alto'] = fuzz.trimf(
    riesgo.universe,
    [60,100,100]
)

# =====================================================
# REGLAS DIFUSAS
# =====================================================

regla1 = ctrl.Rule(
    lluvia_fz['alta'] &
    humedad_fz['saturada'] &
    drenaje_fz['bajo'],
    riesgo['alto']
)

regla2 = ctrl.Rule(
    lluvia_fz['media'] &
    drenaje_fz['medio'],
    riesgo['medio']
)

regla3 = ctrl.Rule(
    lluvia_fz['baja'] &
    drenaje_fz['alto'],
    riesgo['bajo']
)

regla4 = ctrl.Rule(
    pendiente_fz['llana'] &
    lluvia_fz['alta'],
    riesgo['alto']
)

# =====================================================
# SISTEMA DIFUSO
# =====================================================

sistema = ctrl.ControlSystem([
    regla1,
    regla2,
    regla3,
    regla4
])

simulador = ctrl.ControlSystemSimulation(
    sistema
)

# =====================================================
# INPUTS
# =====================================================

simulador.input['lluvia'] = lluvia
simulador.input['humedad'] = humedad
simulador.input['drenaje'] = drenaje
simulador.input['pendiente'] = pendiente

# =====================================================
# EJECUTAR SISTEMA
# =====================================================

simulador.compute()

riesgo_final = simulador.output['riesgo']

# =====================================================
# RESULTADOS
# =====================================================

st.subheader("📊 Resultados")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Predicción ML",
        prediccion_ml[0]
    )

with col2:
    st.metric(
        "Riesgo Difuso",
        round(riesgo_final,2)
    )

# =====================================================
# ALERTAS
# =====================================================

if riesgo_final >= 80:

    st.error(
        "🚨 RIESGO CRÍTICO DE INUNDACIÓN"
    )

elif riesgo_final >= 60:

    st.warning(
        "⚠️ RIESGO ALTO"
    )

elif riesgo_final >= 40:

    st.info(
        "🟡 RIESGO MEDIO"
    )

else:

    st.success(
        "🟢 RIESGO BAJO"
    )

# =====================================================
# INTERPRETACION
# =====================================================

st.subheader("🧠 Interpretación")

st.write("""
El modelo de Machine Learning aprende patrones asociados
al riesgo de inundación a partir de datos hidrometeorológicos.

Posteriormente, el sistema de lógica difusa interpreta
escenarios graduales e inciertos para generar un riesgo final
más interpretable e inteligente.
""")
# =========================================================
# VISUALIZACION AVANZADA ML + LOGICA DIFUSA
# PEGAR DEBAJO DE LOS RESULTADOS
# =========================================================

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# =========================================================
# TITULO
# =========================================================

st.markdown("---")
st.header("📈 Visualización Inteligente del Sistema")

# =========================================================
# VALORES INGRESADOS
# =========================================================

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

st.dataframe(
    df_inputs,
    use_container_width=True
)

# =========================================================
# GRAFICO VARIABLES INGRESADAS
# =========================================================

st.subheader("📊 Variables climáticas")

fig_bar, ax = plt.subplots(figsize=(8,4))

ax.bar(
    df_inputs["Variable"],
    df_inputs["Valor"]
)

ax.set_ylabel("Valor")
ax.set_title("Valores ingresados")

st.pyplot(fig_bar)

# =========================================================
# IMPORTANCIA DE VARIABLES ML
# =========================================================

st.subheader("🧠 Importancia de Variables - Machine Learning")

importancias = modelo.feature_importances_

df_importancia = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": importancias
})

df_importancia = df_importancia.sort_values(
    by="Importancia",
    ascending=True
)

fig_imp, ax = plt.subplots(figsize=(8,4))

ax.barh(
    df_importancia["Variable"],
    df_importancia["Importancia"]
)

ax.set_title("Importancia de Variables")
ax.set_xlabel("Importancia")

st.pyplot(fig_imp)

# =========================================================
# EXPLICACION ML
# =========================================================

st.info(f"""
El modelo de Machine Learning analizó las variables ingresadas y
detectó patrones similares a escenarios de riesgo:

- 🌧️ Lluvia: {lluvia} mm
- 💧 Humedad: {humedad} %
- 🚰 Drenaje: {drenaje} %
- ⛰️ Pendiente: {pendiente} %

Predicción ML:
➡️ {prediccion_ml[0]}
""")

# =========================================================
# MEDIDOR LOGICA DIFUSA
# =========================================================

st.subheader("🌫️ Inferencia Difusa")

fig_gauge = go.Figure(go.Indicator(

    mode = "gauge+number",

    value = riesgo_final,

    title = {
        'text': "Nivel de Riesgo Difuso"
    },

    gauge = {
        'axis': {
            'range': [0,100]
        },

        'steps': [

            {
                'range': [0,40],
                'color': "green"
            },

            {
                'range': [40,70],
                'color': "yellow"
            },

            {
                'range': [70,100],
                'color': "red"
            }
        ]
    }
))

st.plotly_chart(
    fig_gauge,
    use_container_width=True
)

# =========================================================
# VISUALIZACION DIFUSA
# =========================================================

st.subheader("📉 Activación de Reglas Difusas")

riesgo.view(sim=simulador)

st.pyplot(plt.gcf())

# =========================================================
# INTERPRETACION FINAL
# =========================================================

st.subheader("🧾 Interpretación Inteligente")

if riesgo_final >= 80:

    interpretacion = """
    El sistema detecta un escenario crítico de inundación.
    
    Existe alta probabilidad de acumulación de agua debido
    a precipitaciones elevadas, drenaje insuficiente y
    condiciones favorables para anegamientos.
    """

elif riesgo_final >= 60:

    interpretacion = """
    El sistema detecta un riesgo alto de inundación.
    
    Las variables climáticas presentan condiciones
    potencialmente peligrosas.
    """

elif riesgo_final >= 40:

    interpretacion = """
    El sistema detecta un riesgo moderado.
    
    Se recomienda monitoreo preventivo.
    """

else:

    interpretacion = """
    El sistema detecta un riesgo bajo de inundación.
    
    Las condiciones actuales no representan peligro significativo.
    """

st.write(interpretacion)

# =========================================================
# COMPARACION ML VS DIFUSO
# =========================================================

st.subheader("⚖️ Comparación ML vs Lógica Difusa")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Predicción ML",
        prediccion_ml[0]
    )

with col2:

    st.metric(
        "Valor Difuso",
        round(riesgo_final,2)
    )

# =========================================================
# RESUMEN TECNICO
# =========================================================

st.markdown("---")

st.success("""
✅ El modelo Machine Learning aprende patrones de inundación
a partir de datos históricos simulados.

✅ La lógica difusa interpreta escenarios graduales e inciertos.

✅ El sistema híbrido combina predicción automática +
razonamiento inteligente.
""")
