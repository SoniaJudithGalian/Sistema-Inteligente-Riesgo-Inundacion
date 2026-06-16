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
# ESTILO VISUAL MODERNO
# =====================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background-color: #E2E8F0;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 18px;
    color: #475569;
    margin-bottom: 25px;
}

.card {
    background-color: white;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0px 6px 20px rgba(15, 23, 42, 0.08);
    border: 1px solid #E2E8F0;
    margin-bottom: 18px;
}

.card-title {
    font-size: 15px;
    color: #64748B;
    font-weight: 600;
    margin-bottom: 8px;
}

.card-value {
    font-size: 30px;
    color: #0F172A;
    font-weight: 800;
}

.card-small {
    font-size: 14px;
    color: #64748B;
}

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #0F172A;
    margin-top: 20px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# CARGAR MODELO
# =====================================================

modelo = joblib.load("modelo_ml.pkl")

# =====================================================
# TITULO
# =====================================================

st.markdown('<p class="main-title">🌊 Sistema Inteligente de Riesgo de Inundaciones</p>', unsafe_allow_html=True)

st.markdown("""
<p class="subtitle">
Sistema híbrido que combina Machine Learning y Lógica Difusa para estimar el riesgo de inundación
a partir de variables climáticas y territoriales.
</p>
""", unsafe_allow_html=True)

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
# INTERPRETAR CLASE ML
# =====================================================

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

regla5 = ctrl.Rule(
    lluvia_fz['alta'],
    riesgo['alto']
)

regla6 = ctrl.Rule(
    lluvia_fz['media'],
    riesgo['medio']
)

regla7 = ctrl.Rule(
    lluvia_fz['baja'],
    riesgo['bajo']
)

regla8 = ctrl.Rule(
    (lluvia_fz['media'] | lluvia_fz['alta']) &
    drenaje_fz['bajo'],
    riesgo['alto']
)


regla9 = ctrl.Rule(
    drenaje_fz['alto'],
    riesgo['bajo']
)

# =====================================================
# SISTEMA DIFUSO
# =====================================================

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
riesgo_final = simulador.output.get('riesgo', 0)

# =====================================================
# RESULTADOS - DISEÑO MODERNO
# =====================================================

import plotly.graph_objects as go
import plotly.express as px

# Paleta profesional
COLOR_AZUL = "#0F172A"
COLOR_CELESTE = "#38BDF8"
COLOR_VERDE = "#22C55E"
COLOR_AMARILLO = "#FACC15"
COLOR_NARANJA = "#F97316"
COLOR_ROJO = "#EF4444"
COLOR_GRIS = "#64748B"


def obtener_nivel_riesgo(valor):
    if valor >= 80:
        return "Crítico", "🚨", COLOR_ROJO
    elif valor >= 60:
        return "Alto", "⚠️", COLOR_NARANJA
    elif valor >= 40:
        return "Medio", "🟡", COLOR_AMARILLO
    else:
        return "Bajo", "🟢", COLOR_VERDE


nivel_difuso, icono_difuso, color_difuso = obtener_nivel_riesgo(riesgo_final)


st.markdown("---")
st.markdown('<p class="section-title">📊 Panel de Resultados</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Predicción Machine Learning</div>
        <div class="card-value">{prediccion_texto}</div>
        <div class="card-small">Clasificación generada por el modelo entrenado</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Riesgo Difuso</div>
        <div class="card-value">{round(riesgo_final, 2)} / 100</div>
        <div class="card-small">Resultado del sistema de lógica difusa</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Nivel Interpretado</div>
        <div class="card-value">{icono_difuso} {nivel_difuso}</div>
        <div class="card-small">Lectura final del escenario analizado</div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# ALERTA VISUAL
# =====================================================

if riesgo_final >= 80:
    st.error("🚨 RIESGO CRÍTICO DE INUNDACIÓN: se recomienda intervención inmediata.")
elif riesgo_final >= 60:
    st.warning("⚠️ RIESGO ALTO: se recomienda monitoreo y prevención.")
elif riesgo_final >= 40:
    st.info("🟡 RIESGO MEDIO: se recomienda seguimiento preventivo.")
else:
    st.success("🟢 RIESGO BAJO: las condiciones actuales no presentan peligro significativo.")


# =====================================================
# DATAFRAME DE VARIABLES
# =====================================================

df_inputs = pd.DataFrame({
    "Variable": [
        "Lluvia",
        "Humedad del suelo",
        "Capacidad de drenaje",
        "Pendiente topográfica"
    ],
    "Valor": [
        lluvia,
        humedad,
        drenaje,
        pendiente
    ],
    "Máximo": [
        120,
        100,
        100,
        10
    ],
    "Unidad": [
        "mm",
        "%",
        "%",
        "%"
    ]
})

df_inputs["Porcentaje"] = round((df_inputs["Valor"] / df_inputs["Máximo"]) * 100, 2)
df_inputs["Etiqueta"] = df_inputs["Valor"].astype(str) + " " + df_inputs["Unidad"]


# =====================================================
# GRAFICOS DE ENTRADA
# =====================================================

st.markdown('<p class="section-title">📥 Variables de Entrada</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 1])

with col1:
    fig_barras = px.bar(
        df_inputs,
        x="Porcentaje",
        y="Variable",
        orientation="h",
        text="Etiqueta",
        color="Variable",
        color_discrete_sequence=[
            "#38BDF8",
            "#0EA5E9",
            "#22C55E",
            "#A855F7"
        ]
    )

    fig_barras.update_traces(
        textposition="outside",
        marker_line_width=0
    )

    fig_barras.update_layout(
        title="Valores ingresados normalizados",
        xaxis_title="Porcentaje sobre el valor máximo",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        height=380,
        font=dict(color="#0F172A"),
        xaxis=dict(range=[0, 110])
    )

    st.plotly_chart(fig_barras, use_container_width=True)

with col2:
    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=df_inputs["Porcentaje"],
        theta=df_inputs["Variable"],
        fill="toself",
        name="Valores ingresados",
        line=dict(color=COLOR_CELESTE, width=3)
    ))

    fig_radar.update_layout(
        title="Perfil general del escenario",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        height=380,
        paper_bgcolor="white",
        font=dict(color="#0F172A")
    )

    st.plotly_chart(fig_radar, use_container_width=True)


# =====================================================
# MEDIDOR DE RIESGO DIFUSO
# =====================================================

st.markdown('<p class="section-title">🌫️ Medidor de Riesgo Difuso</p>', unsafe_allow_html=True)

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=riesgo_final,
    number={
        "suffix": " / 100",
        "font": {"size": 36, "color": COLOR_AZUL}
    },
    title={
        "text": f"Nivel de Riesgo: {nivel_difuso}",
        "font": {"size": 22, "color": COLOR_AZUL}
    },
    gauge={
        "axis": {
            "range": [0, 100],
            "tickwidth": 1,
            "tickcolor": COLOR_GRIS
        },
        "bar": {
            "color": color_difuso
        },
        "bgcolor": "white",
        "borderwidth": 1,
        "bordercolor": "#CBD5E1",
        "steps": [
            {"range": [0, 40], "color": "#DCFCE7"},
            {"range": [40, 60], "color": "#FEF9C3"},
            {"range": [60, 80], "color": "#FFEDD5"},
            {"range": [80, 100], "color": "#FEE2E2"}
        ],
        "threshold": {
            "line": {"color": COLOR_AZUL, "width": 4},
            "thickness": 0.75,
            "value": riesgo_final
        }
    }
))

fig_gauge.update_layout(
    height=420,
    paper_bgcolor="white",
    font=dict(color=COLOR_AZUL)
)

st.plotly_chart(fig_gauge, use_container_width=True)


# =====================================================
# IMPORTANCIA DE VARIABLES ML
# =====================================================

st.markdown('<p class="section-title">🧠 Importancia de Variables del Modelo ML</p>', unsafe_allow_html=True)

modelo_base = modelo

if hasattr(modelo, "named_steps"):
    modelo_base = list(modelo.named_steps.values())[-1]

if hasattr(modelo_base, "feature_importances_"):

    importancias = modelo_base.feature_importances_

    df_importancia = pd.DataFrame({
        "Variable": [
            "Lluvia",
            "Humedad",
            "Drenaje",
            "Pendiente"
        ],
        "Importancia": importancias
    })

    df_importancia = df_importancia.sort_values(
        by="Importancia",
        ascending=True
    )

    fig_imp = px.bar(
        df_importancia,
        x="Importancia",
        y="Variable",
        orientation="h",
        text=round(df_importancia["Importancia"], 3),
        color="Importancia",
        color_continuous_scale="Blues"
    )

    fig_imp.update_traces(
        textposition="outside"
    )

    fig_imp.update_layout(
        title="Variables más influyentes según el modelo",
        xaxis_title="Importancia",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380,
        font=dict(color="#0F172A"),
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.info(
        "El modelo cargado no tiene atributo feature_importances_. "
        "Esto puede pasar si usás Regresión Logística u otro modelo que no calcula importancia de variables directamente."
    )


# =====================================================
# FUNCIONES DE PERTENENCIA DIFUSA
# =====================================================

st.markdown('<p class="section-title">📉 Funciones de Pertenencia Difusa</p>', unsafe_allow_html=True)

def graficar_pertenencia(variable_fz, valor_actual, titulo, unidad):
    fig = go.Figure()

    colores = [
        "#22C55E",
        "#FACC15",
        "#EF4444",
        "#38BDF8"
    ]

    for i, nombre in enumerate(variable_fz.terms):
        fig.add_trace(go.Scatter(
            x=variable_fz.universe,
            y=variable_fz[nombre].mf,
            mode="lines",
            name=nombre.capitalize(),
            line=dict(width=3, color=colores[i % len(colores)])
        ))

    fig.add_vline(
        x=valor_actual,
        line_width=3,
        line_dash="dash",
        line_color="#0F172A",
        annotation_text=f"Valor actual: {valor_actual} {unidad}",
        annotation_position="top"
    )

    fig.update_layout(
        title=titulo,
        xaxis_title=unidad,
        yaxis_title="Grado de pertenencia",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        font=dict(color="#0F172A"),
        legend_title="Categorías"
    )

    return fig


tab1, tab2, tab3, tab4 = st.tabs([
    "🌧️ Lluvia",
    "💧 Humedad",
    "🚰 Drenaje",
    "⛰️ Pendiente"
])

with tab1:
    st.plotly_chart(
        graficar_pertenencia(lluvia_fz, lluvia, "Pertenencia difusa de la lluvia", "mm"),
        use_container_width=True
    )

with tab2:
    st.plotly_chart(
        graficar_pertenencia(humedad_fz, humedad, "Pertenencia difusa de la humedad", "%"),
        use_container_width=True
    )

with tab3:
    st.plotly_chart(
        graficar_pertenencia(drenaje_fz, drenaje, "Pertenencia difusa del drenaje", "%"),
        use_container_width=True
    )

with tab4:
    st.plotly_chart(
        graficar_pertenencia(pendiente_fz, pendiente, "Pertenencia difusa de la pendiente", "%"),
        use_container_width=True
    )


# =====================================================
# INTERPRETACION INTELIGENTE
# =====================================================

st.markdown('<p class="section-title">🧾 Interpretación Inteligente</p>', unsafe_allow_html=True)

if riesgo_final >= 80:
    interpretacion = """
    El sistema detecta un escenario crítico. Las condiciones ingresadas indican una alta posibilidad
    de acumulación de agua y anegamientos. Se recomienda activar protocolos de emergencia,
    monitorear zonas vulnerables y revisar la capacidad de drenaje.
    """
elif riesgo_final >= 60:
    interpretacion = """
    El sistema detecta un riesgo alto. Las variables muestran condiciones que podrían generar
    problemas de escurrimiento o acumulación de agua. Se recomienda monitoreo preventivo.
    """
elif riesgo_final >= 40:
    interpretacion = """
    El sistema detecta un riesgo medio. No se observa un escenario extremo, pero existen señales
    que requieren seguimiento, especialmente si aumenta la lluvia o baja la capacidad de drenaje.
    """
else:
    interpretacion = """
    El sistema detecta un riesgo bajo. Las condiciones actuales no representan peligro significativo,
    aunque se recomienda mantener el monitoreo si el clima cambia.
    """

st.markdown(f"""
<div class="card">
    <div class="card-title">Lectura del sistema</div>
    <div style="font-size:17px; color:#334155; line-height:1.6;">
        {interpretacion}
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# RECOMENDACIONES AUTOMATICAS
# =====================================================

st.markdown('<p class="section-title">✅ Recomendaciones Automáticas</p>', unsafe_allow_html=True)

recomendaciones = []

if lluvia >= 80:
    recomendaciones.append("🌧️ Monitorear lluvias intensas y acumulación de agua en zonas bajas.")

if humedad >= 80:
    recomendaciones.append("💧 El suelo se encuentra muy húmedo o saturado; puede reducir la absorción de agua.")

if drenaje <= 30:
    recomendaciones.append("🚰 Revisar desagües, canales y bocas de tormenta por posible baja capacidad de drenaje.")

if pendiente <= 3:
    recomendaciones.append("⛰️ La pendiente baja favorece la acumulación de agua; controlar zonas llanas.")

if len(recomendaciones) == 0:
    recomendaciones.append("🟢 No se detectan factores críticos importantes en las variables ingresadas.")

for rec in recomendaciones:
    st.write(rec)


# =====================================================
# CIERRE TECNICO
# =====================================================

st.markdown("---")

st.success(
    "El sistema híbrido combina la capacidad predictiva del Machine Learning "
    "con la interpretación gradual de la lógica difusa, permitiendo una lectura "
    "más clara, visual e interpretable del riesgo de inundación."
)
