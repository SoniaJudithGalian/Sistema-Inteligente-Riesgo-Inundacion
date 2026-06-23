import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

import skfuzzy as fuzz
from skfuzzy import control as ctrl

import plotly.express as px
import plotly.graph_objects as go

import folium
from streamlit_folium import st_folium


# =====================================================
# CONFIGURACION GENERAL
# =====================================================

st.set_page_config(
    page_title="Sistema Inteligente de Riesgo de Inundaciones",
    layout="wide"
)


# =====================================================
# ESTILO VISUAL PROFESIONAL
# =====================================================

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC;
    }

    [data-testid="stSidebar"] {
        background-color: #E2E8F0;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }

    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 6px;
        line-height: 1.15;
    }

    .subtitle {
        font-size: 18px;
        color: #475569;
        margin-bottom: 25px;
        line-height: 1.55;
    }

    .section-title {
        font-size: 25px;
        font-weight: 800;
        color: #0F172A;
        margin-top: 28px;
        margin-bottom: 14px;
    }

    .card {
        background-color: white;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0px 8px 22px rgba(15, 23, 42, 0.07);
        margin-bottom: 16px;
    }

    .card-title {
        font-size: 14px;
        color: #64748B;
        font-weight: 700;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .card-value {
        font-size: 28px;
        color: #0F172A;
        font-weight: 800;
        margin-bottom: 4px;
        line-height: 1.15;
    }

    .card-small {
        font-size: 14px;
        color: #64748B;
        line-height: 1.5;
    }

    .info-box {
        background-color: #EFF6FF;
        padding: 18px;
        border-radius: 14px;
        border-left: 5px solid #2563EB;
        color: #1E3A8A;
        font-size: 16px;
        line-height: 1.55;
        margin-bottom: 16px;
    }

    .risk-low {
        background-color: #DCFCE7;
        border-left: 6px solid #16A34A;
        padding: 18px;
        border-radius: 14px;
        color: #14532D;
        font-weight: 600;
        line-height: 1.55;
    }

    .risk-medium {
        background-color: #FEF9C3;
        border-left: 6px solid #CA8A04;
        padding: 18px;
        border-radius: 14px;
        color: #713F12;
        font-weight: 600;
        line-height: 1.55;
    }

    .risk-high {
        background-color: #FFEDD5;
        border-left: 6px solid #EA580C;
        padding: 18px;
        border-radius: 14px;
        color: #7C2D12;
        font-weight: 600;
        line-height: 1.55;
    }

    .risk-critical {
        background-color: #FEE2E2;
        border-left: 6px solid #DC2626;
        padding: 18px;
        border-radius: 14px;
        color: #7F1D1D;
        font-weight: 600;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# COLORES
# =====================================================

COLOR_AZUL = "#0F172A"
COLOR_GRIS = "#64748B"
COLOR_CELESTE = "#38BDF8"
COLOR_VERDE = "#22C55E"
COLOR_AMARILLO = "#FACC15"
COLOR_NARANJA = "#F97316"
COLOR_ROJO = "#EF4444"


# =====================================================
# CIUDADES
# =====================================================

CIUDADES = {
    "Neuquén Capital": {
        "lat": -38.9516,
        "lon": -68.0591
    },
    "Bahía Blanca": {
        "lat": -38.7183,
        "lon": -62.2663
    },
    "Buenos Aires": {
        "lat": -34.6037,
        "lon": -58.3816
    },
    "Córdoba": {
        "lat": -31.4201,
        "lon": -64.1888
    },
    "Rosario": {
        "lat": -32.9442,
        "lon": -60.6505
    },
    "Mendoza": {
        "lat": -32.8895,
        "lon": -68.8458
    },
    "La Plata": {
        "lat": -34.9215,
        "lon": -57.9545
    },
    "Mar del Plata": {
        "lat": -38.0055,
        "lon": -57.5426
    },
    "Salta": {
        "lat": -24.7821,
        "lon": -65.4232
    },
    "San Miguel de Tucumán": {
        "lat": -26.8083,
        "lon": -65.2176
    }
}


CASO_HISTORICO_BAHIA = {
    "ciudad": "Bahía Blanca",
    "fecha_inicio": "2025-03-04",
    "fecha_fin": "2025-03-08",
    "fecha_evento_principal": "2025-03-07",
    "lluvia_acumulada_documentada_mm": 312,
    "lluvia_evento_principal_mm": 290,
    "humedad_referencial": 95,
    "descripcion": (
        "Evento histórico de inundación ocurrido en Bahía Blanca durante marzo de 2025. "
        "Se analiza el período completo del 4 al 8 de marzo, con foco en el evento extremo "
        "registrado el 7 de marzo."
    )
}


# =====================================================
# API OPEN-METEO
# =====================================================

@st.cache_data(ttl=1800)
def obtener_clima_actual_open_meteo(latitud, longitud):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitud,
        "longitude": longitud,
        "hourly": "precipitation,relative_humidity_2m",
        "forecast_days": 1,
        "timezone": "America/Argentina/Buenos_Aires"
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()
    df_clima = pd.DataFrame(data["hourly"])

    lluvia_24h = float(df_clima["precipitation"].sum())
    humedad_promedio = float(df_clima["relative_humidity_2m"].mean())

    return lluvia_24h, humedad_promedio, df_clima


@st.cache_data(ttl=3600)
def obtener_clima_historico_open_meteo(latitud, longitud, fecha_inicio, fecha_fin):
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitud,
        "longitude": longitud,
        "start_date": fecha_inicio,
        "end_date": fecha_fin,
        "hourly": "precipitation,relative_humidity_2m",
        "timezone": "America/Argentina/Buenos_Aires"
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()
    df_historico = pd.DataFrame(data["hourly"])

    df_historico["time"] = pd.to_datetime(df_historico["time"])
    df_historico["fecha"] = df_historico["time"].dt.date

    df_diario = (
        df_historico
        .groupby("fecha", as_index=False)
        .agg(
            precipitacion_mm=("precipitation", "sum"),
            humedad_promedio=("relative_humidity_2m", "mean")
        )
    )

    df_diario["precipitacion_mm"] = df_diario["precipitacion_mm"].round(2)
    df_diario["humedad_promedio"] = df_diario["humedad_promedio"].round(2)

    dias_con_lluvia = df_diario[df_diario["precipitacion_mm"] > 0].copy()

    lluvia_total_api = float(df_diario["precipitacion_mm"].sum())
    humedad_promedio_api = float(df_diario["humedad_promedio"].mean())

    return lluvia_total_api, humedad_promedio_api, df_historico, df_diario, dias_con_lluvia


# =====================================================
# MODELO ML
# =====================================================

@st.cache_resource
def cargar_modelo_ml():
    try:
        modelo = joblib.load("modelo_ml.pkl")
        return modelo
    except Exception:
        return None


def interpretar_prediccion_ml(prediccion):
    valor = prediccion[0]

    mapa_numerico = {
        0: "Bajo",
        1: "Medio",
        2: "Alto",
        3: "Crítico"
    }

    if isinstance(valor, (int, np.integer)):
        return mapa_numerico.get(int(valor), "No disponible")

    if isinstance(valor, (float, np.floating)):
        return mapa_numerico.get(int(valor), "No disponible")

    texto = str(valor).strip().lower()

    if texto in ["bajo", "low"]:
        return "Bajo"
    if texto in ["medio", "moderado", "medium"]:
        return "Medio"
    if texto in ["alto", "high"]:
        return "Alto"
    if texto in ["critico", "crítico", "critical"]:
        return "Crítico"

    return str(valor)


# =====================================================
# LOGICA DIFUSA CORREGIDA
# =====================================================

def crear_sistema_difuso():
    lluvia_fz = ctrl.Antecedent(np.arange(0, 351, 1), "lluvia")
    humedad_fz = ctrl.Antecedent(np.arange(0, 101, 1), "humedad")
    drenaje_fz = ctrl.Antecedent(np.arange(0, 101, 1), "drenaje")
    pendiente_fz = ctrl.Antecedent(np.arange(0, 11, 1), "pendiente")
    riesgo_fz = ctrl.Consequent(np.arange(0, 101, 1), "riesgo")

    lluvia_fz["baja"] = fuzz.trimf(lluvia_fz.universe, [0, 0, 50])
    lluvia_fz["media"] = fuzz.trimf(lluvia_fz.universe, [30, 80, 140])
    lluvia_fz["alta"] = fuzz.trimf(lluvia_fz.universe, [100, 170, 250])
    lluvia_fz["extrema"] = fuzz.trimf(lluvia_fz.universe, [220, 350, 350])

    humedad_fz["baja"] = fuzz.trimf(humedad_fz.universe, [0, 0, 45])
    humedad_fz["media"] = fuzz.trimf(humedad_fz.universe, [30, 60, 85])
    humedad_fz["alta"] = fuzz.trimf(humedad_fz.universe, [70, 100, 100])

    drenaje_fz["bajo"] = fuzz.trimf(drenaje_fz.universe, [0, 0, 45])
    drenaje_fz["medio"] = fuzz.trimf(drenaje_fz.universe, [30, 55, 80])
    drenaje_fz["alto"] = fuzz.trimf(drenaje_fz.universe, [65, 100, 100])

    pendiente_fz["llana"] = fuzz.trimf(pendiente_fz.universe, [0, 0, 3])
    pendiente_fz["moderada"] = fuzz.trimf(pendiente_fz.universe, [2, 5, 8])
    pendiente_fz["alta"] = fuzz.trimf(pendiente_fz.universe, [6, 10, 10])

    riesgo_fz["bajo"] = fuzz.trimf(riesgo_fz.universe, [0, 10, 35])
    riesgo_fz["medio"] = fuzz.trimf(riesgo_fz.universe, [25, 50, 70])
    riesgo_fz["alto"] = fuzz.trimf(riesgo_fz.universe, [60, 75, 90])
    riesgo_fz["critico"] = fuzz.trimf(riesgo_fz.universe, [80, 100, 100])

    reglas = [
        ctrl.Rule(lluvia_fz["baja"] & drenaje_fz["alto"], riesgo_fz["bajo"]),
        ctrl.Rule(lluvia_fz["baja"] & drenaje_fz["medio"], riesgo_fz["bajo"]),
        ctrl.Rule(lluvia_fz["baja"] & drenaje_fz["bajo"], riesgo_fz["medio"]),
        ctrl.Rule(lluvia_fz["baja"] & humedad_fz["alta"] & drenaje_fz["bajo"], riesgo_fz["medio"]),

        ctrl.Rule(lluvia_fz["media"] & drenaje_fz["alto"], riesgo_fz["medio"]),
        ctrl.Rule(lluvia_fz["media"] & drenaje_fz["medio"], riesgo_fz["medio"]),
        ctrl.Rule(lluvia_fz["media"] & drenaje_fz["bajo"], riesgo_fz["alto"]),
        ctrl.Rule(lluvia_fz["media"] & humedad_fz["alta"], riesgo_fz["alto"]),
        ctrl.Rule(lluvia_fz["media"] & pendiente_fz["llana"], riesgo_fz["medio"]),

        ctrl.Rule(lluvia_fz["alta"] & drenaje_fz["alto"], riesgo_fz["alto"]),
        ctrl.Rule(lluvia_fz["alta"] & drenaje_fz["medio"], riesgo_fz["alto"]),
        ctrl.Rule(lluvia_fz["alta"] & drenaje_fz["bajo"], riesgo_fz["critico"]),
        ctrl.Rule(lluvia_fz["alta"] & humedad_fz["alta"], riesgo_fz["alto"]),
        ctrl.Rule(lluvia_fz["alta"] & pendiente_fz["llana"], riesgo_fz["alto"]),

        ctrl.Rule(lluvia_fz["extrema"], riesgo_fz["critico"]),
        ctrl.Rule(lluvia_fz["extrema"] & humedad_fz["alta"], riesgo_fz["critico"]),
        ctrl.Rule(lluvia_fz["extrema"] & drenaje_fz["bajo"], riesgo_fz["critico"]),
        ctrl.Rule(lluvia_fz["extrema"] & pendiente_fz["llana"], riesgo_fz["critico"]),

        ctrl.Rule(humedad_fz["alta"] & drenaje_fz["bajo"], riesgo_fz["alto"]),
        ctrl.Rule(pendiente_fz["llana"] & drenaje_fz["bajo"] & humedad_fz["alta"], riesgo_fz["alto"]),
        ctrl.Rule(drenaje_fz["alto"] & lluvia_fz["baja"], riesgo_fz["bajo"]),
    ]

    sistema = ctrl.ControlSystem(reglas)

    return sistema, lluvia_fz, humedad_fz, drenaje_fz, pendiente_fz, riesgo_fz


def calcular_riesgo_difuso(lluvia, humedad, drenaje, pendiente):
    lluvia = float(np.clip(lluvia, 0, 350))
    humedad = float(np.clip(humedad, 0, 100))
    drenaje = float(np.clip(drenaje, 0, 100))
    pendiente = float(np.clip(pendiente, 0, 10))

    sistema, lluvia_fz, humedad_fz, drenaje_fz, pendiente_fz, riesgo_fz = crear_sistema_difuso()

    simulador = ctrl.ControlSystemSimulation(sistema)

    simulador.input["lluvia"] = lluvia
    simulador.input["humedad"] = humedad
    simulador.input["drenaje"] = drenaje
    simulador.input["pendiente"] = pendiente

    try:
        simulador.compute()
        riesgo_final = float(simulador.output["riesgo"])
    except Exception:
        riesgo_final = 10.0

    return riesgo_final, lluvia_fz, humedad_fz, drenaje_fz, pendiente_fz, riesgo_fz


def obtener_nivel_riesgo(valor):
    if valor >= 80:
        return "Crítico", COLOR_ROJO, "risk-critical"
    elif valor >= 60:
        return "Alto", COLOR_NARANJA, "risk-high"
    elif valor >= 40:
        return "Medio", COLOR_AMARILLO, "risk-medium"
    else:
        return "Bajo", COLOR_VERDE, "risk-low"


# =====================================================
# GRAFICOS Y MAPA
# =====================================================

def crear_mapa_satelital(latitud, longitud, ciudad, riesgo_final, nivel_riesgo, color_riesgo):
    mapa = folium.Map(
        location=[latitud, longitud],
        zoom_start=12,
        tiles=None
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Vista satelital",
        overlay=False,
        control=True
    ).add_to(mapa)

    radio = 350 + riesgo_final * 18

    folium.Circle(
        location=[latitud, longitud],
        radius=radio,
        color=color_riesgo,
        fill=True,
        fill_color=color_riesgo,
        fill_opacity=0.35,
        popup=f"{ciudad} - Riesgo {nivel_riesgo} - {riesgo_final:.2f}/100"
    ).add_to(mapa)

    folium.Marker(
        location=[latitud, longitud],
        tooltip=f"{ciudad} - Riesgo {nivel_riesgo}",
        popup=f"Riesgo estimado: {riesgo_final:.2f}/100"
    ).add_to(mapa)

    return mapa


def graficar_gauge(riesgo_final, nivel_riesgo, color_riesgo):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=riesgo_final,
            number={
                "suffix": " / 100",
                "font": {"size": 38, "color": COLOR_AZUL}
            },
            title={
                "text": f"Riesgo difuso: {nivel_riesgo}",
                "font": {"size": 24, "color": COLOR_AZUL}
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": COLOR_GRIS
                },
                "bar": {
                    "color": color_riesgo,
                    "thickness": 0.22
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
                    "line": {"color": COLOR_AZUL, "width": 5},
                    "thickness": 0.85,
                    "value": riesgo_final
                }
            }
        )
    )

    fig.update_layout(
        height=430,
        paper_bgcolor="white",
        margin=dict(l=30, r=30, t=70, b=20),
        font=dict(color=COLOR_AZUL)
    )

    return fig


def graficar_serie_climatica(df_clima):
    df = df_clima.copy()
    df["time"] = pd.to_datetime(df["time"])

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["time"],
            y=df["precipitation"],
            name="Precipitación horaria",
            yaxis="y1"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["relative_humidity_2m"],
            mode="lines+markers",
            name="Humedad relativa",
            yaxis="y2"
        )
    )

    fig.update_layout(
        title="Serie horaria de precipitación y humedad",
        xaxis_title="Fecha y hora",
        yaxis=dict(
            title="Precipitación horaria en mm",
            side="left"
        ),
        yaxis2=dict(
            title="Humedad relativa en %",
            overlaying="y",
            side="right",
            range=[0, 100]
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=430,
        font=dict(color=COLOR_AZUL),
        legend=dict(orientation="h", y=-0.25)
    )

    return fig


def graficar_precipitacion_diaria(df_diario):
    fig = px.bar(
        df_diario,
        x="fecha",
        y="precipitacion_mm",
        text="precipitacion_mm",
        title="Precipitación diaria durante el período histórico",
        labels={
            "fecha": "Fecha",
            "precipitacion_mm": "Precipitación diaria en mm"
        }
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380,
        font=dict(color=COLOR_AZUL)
    )

    return fig


def graficar_variables(df_inputs):
    fig = px.bar(
        df_inputs,
        x="Porcentaje",
        y="Variable",
        orientation="h",
        text="Etiqueta",
        color="Variable",
        color_discrete_sequence=[
            "#38BDF8",
            "#2563EB",
            "#22C55E",
            "#A855F7"
        ]
    )

    fig.update_traces(
        textposition="outside",
        marker_line_width=0
    )

    fig.update_layout(
        title="Variables normalizadas del escenario analizado",
        xaxis_title="Porcentaje sobre el valor máximo de referencia",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        height=390,
        font=dict(color=COLOR_AZUL),
        xaxis=dict(range=[0, 110])
    )

    return fig


def graficar_pertenencia(variable_fz, valor_actual, titulo, unidad):
    fig = go.Figure()

    colores = [
        COLOR_VERDE,
        COLOR_AMARILLO,
        COLOR_NARANJA,
        COLOR_ROJO,
        COLOR_CELESTE
    ]

    for i, nombre in enumerate(variable_fz.terms):
        fig.add_trace(
            go.Scatter(
                x=variable_fz.universe,
                y=variable_fz[nombre].mf,
                mode="lines",
                name=nombre.capitalize(),
                line=dict(width=3, color=colores[i % len(colores)])
            )
        )

    fig.add_vline(
        x=valor_actual,
        line_width=3,
        line_dash="dash",
        line_color=COLOR_AZUL,
        annotation_text=f"Valor actual: {valor_actual} {unidad}",
        annotation_position="top"
    )

    fig.update_layout(
        title=titulo,
        xaxis_title=unidad,
        yaxis_title="Grado de pertenencia",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=360,
        font=dict(color=COLOR_AZUL),
        legend_title="Categorías"
    )

    return fig


# =====================================================
# INTERPRETACION Y RECOMENDACIONES
# =====================================================

def obtener_interpretacion(riesgo_final):
    if riesgo_final >= 80:
        return (
            "El sistema identifica un escenario crítico. La combinación de precipitación intensa, "
            "alta humedad, baja capacidad de drenaje o pendiente reducida puede favorecer la acumulación "
            "rápida de agua. Se recomienda activar protocolos de emergencia, monitorear zonas vulnerables "
            "y priorizar la revisión de desagües, canales y sectores bajos."
        )

    if riesgo_final >= 60:
        return (
            "El sistema identifica un escenario de riesgo alto. Las condiciones analizadas podrían generar "
            "anegamientos, problemas de escurrimiento o saturación del suelo. Se recomienda monitoreo preventivo "
            "y seguimiento de la evolución climática."
        )

    if riesgo_final >= 40:
        return (
            "El sistema identifica un escenario de riesgo medio. No se observa una situación extrema, pero existen "
            "variables que requieren atención, especialmente si aumenta la lluvia o disminuye la capacidad de drenaje."
        )

    return (
        "El sistema identifica un escenario de riesgo bajo. Las condiciones actuales no muestran señales críticas, "
        "aunque se recomienda mantener el monitoreo ante posibles cambios climáticos."
    )


def generar_recomendaciones(lluvia, humedad, drenaje, pendiente, riesgo_final):
    recomendaciones = []

    if lluvia >= 200:
        recomendaciones.append("La precipitación acumulada corresponde a un evento extremo. Se recomienda activar protocolos de emergencia y seguimiento permanente.")
    elif lluvia >= 90:
        recomendaciones.append("La precipitación es elevada. Se recomienda controlar zonas bajas, calles anegables y desagües pluviales.")
    elif lluvia >= 50:
        recomendaciones.append("La precipitación es moderada. Se recomienda seguimiento preventivo si la lluvia continúa.")

    if humedad >= 85:
        recomendaciones.append("La humedad es muy alta. El suelo puede tener menor capacidad de absorción y favorecer el escurrimiento superficial.")
    elif humedad >= 70:
        recomendaciones.append("La humedad es elevada. Puede aumentar la sensibilidad del sistema ante nuevas precipitaciones.")

    if drenaje <= 30:
        recomendaciones.append("La capacidad de drenaje ingresada es baja. Se recomienda revisar canales, bocas de tormenta y sistemas pluviales.")
    elif drenaje <= 50:
        recomendaciones.append("La capacidad de drenaje ingresada es media-baja. Conviene realizar monitoreo preventivo.")

    if pendiente <= 3:
        recomendaciones.append("La pendiente ingresada es baja. Las zonas llanas favorecen la acumulación de agua.")

    if riesgo_final >= 80:
        recomendaciones.append("El resultado final indica riesgo crítico. Se recomienda priorizar acciones inmediatas.")
    elif riesgo_final >= 60:
        recomendaciones.append("El resultado final indica riesgo alto. Se recomienda reforzar el seguimiento preventivo.")

    if not recomendaciones:
        recomendaciones.append("No se detectan factores críticos relevantes en el escenario analizado.")

    return recomendaciones


# =====================================================
# TITULO
# =====================================================

st.markdown(
    '<div class="main-title">Sistema Inteligente de Riesgo de Inundaciones</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Aplicación web desarrollada en Streamlit para estimar el riesgo de inundación mediante Machine Learning,
    lógica difusa, datos climáticos externos y visualización geográfica.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    La lluvia y la humedad se obtienen automáticamente desde una API climática o desde el caso histórico.
    La capacidad de drenaje y la pendiente topográfica se ingresan manualmente porque son variables territoriales
    que no provienen directamente de una API meteorológica.
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("## Configuración del análisis")

modo_analisis = st.sidebar.radio(
    "Modo de análisis",
    [
        "Pronóstico actual con API",
        "Caso histórico Bahía Blanca 2025"
    ]
)

df_clima = None
df_diario_historico = None
dias_con_lluvia = None
fuente_datos = ""

if modo_analisis == "Pronóstico actual con API":
    ciudad = st.sidebar.selectbox(
        "Seleccionar ciudad",
        list(CIUDADES.keys()),
        index=0
    )

    latitud = CIUDADES[ciudad]["lat"]
    longitud = CIUDADES[ciudad]["lon"]

    st.sidebar.markdown("### Variables territoriales")
    drenaje = st.sidebar.slider(
        "Capacidad de drenaje estimada (%)",
        min_value=0,
        max_value=100,
        value=50,
        help="Valor manual del prototipo. Un valor bajo indica peor capacidad de escurrimiento."
    )

    pendiente = st.sidebar.slider(
        "Pendiente topográfica estimada (%)",
        min_value=0,
        max_value=10,
        value=3,
        help="Valor manual del prototipo. Una pendiente baja puede favorecer acumulación de agua."
    )

    try:
        lluvia, humedad, df_clima = obtener_clima_actual_open_meteo(
            latitud,
            longitud
        )

        lluvia = round(float(lluvia), 2)
        humedad = round(float(humedad), 2)
        fuente_datos = "Open-Meteo Forecast API"

        st.sidebar.success("Datos climáticos cargados correctamente.")

    except Exception:
        st.sidebar.error("No se pudieron obtener los datos climáticos desde la API.")
        st.sidebar.info("La aplicación se detiene porque la lluvia y la humedad no se ingresan manualmente.")
        st.stop()


else:
    ciudad = CASO_HISTORICO_BAHIA["ciudad"]
    latitud = CIUDADES[ciudad]["lat"]
    longitud = CIUDADES[ciudad]["lon"]

    lluvia = CASO_HISTORICO_BAHIA["lluvia_acumulada_documentada_mm"]
    humedad = CASO_HISTORICO_BAHIA["humedad_referencial"]
    fuente_datos = "Caso histórico documentado y Open-Meteo Archive API"

    st.sidebar.markdown("### Caso histórico")
    st.sidebar.write(f"Ciudad: {ciudad}")
    st.sidebar.write(f"Período: {CASO_HISTORICO_BAHIA['fecha_inicio']} al {CASO_HISTORICO_BAHIA['fecha_fin']}")
    st.sidebar.write(f"Lluvia acumulada documentada: {lluvia} mm")
    st.sidebar.write(f"Evento principal: {CASO_HISTORICO_BAHIA['lluvia_evento_principal_mm']} mm")

    st.sidebar.markdown("### Variables territoriales")
    drenaje = st.sidebar.slider(
        "Capacidad de drenaje estimada (%)",
        min_value=0,
        max_value=100,
        value=20,
        help="Valor manual del prototipo. Para el caso histórico se sugiere probar valores bajos."
    )

    pendiente = st.sidebar.slider(
        "Pendiente topográfica estimada (%)",
        min_value=0,
        max_value=10,
        value=2,
        help="Valor manual del prototipo. En zonas llanas el riesgo de acumulación suele aumentar."
    )

    try:
        lluvia_api_hist, humedad_api_hist, df_clima, df_diario_historico, dias_con_lluvia = obtener_clima_historico_open_meteo(
            latitud,
            longitud,
            CASO_HISTORICO_BAHIA["fecha_inicio"],
            CASO_HISTORICO_BAHIA["fecha_fin"]
        )

        humedad = round(float(humedad_api_hist), 2)

        st.sidebar.success("Datos históricos consultados correctamente.")

    except Exception:
        st.sidebar.warning("No se pudo consultar Open-Meteo histórico. Se mantienen los valores documentados y referenciales.")


st.sidebar.markdown("### Valores utilizados")
st.sidebar.metric("Lluvia acumulada", f"{lluvia:.2f} mm")
st.sidebar.metric("Humedad", f"{humedad:.2f} %")
st.sidebar.metric("Drenaje ingresado", f"{drenaje:.2f} %")
st.sidebar.metric("Pendiente ingresada", f"{pendiente:.2f} %")


# =====================================================
# PREDICCION ML
# =====================================================

modelo = cargar_modelo_ml()

nuevo_caso = pd.DataFrame(
    {
        "lluvia_mm": [lluvia],
        "humedad_suelo": [humedad],
        "capacidad_drenaje": [drenaje],
        "pendiente_topografica": [pendiente]
    }
)

if modelo is not None:
    try:
        prediccion_ml = modelo.predict(nuevo_caso)
        prediccion_ml_texto = interpretar_prediccion_ml(prediccion_ml)
    except Exception:
        prediccion_ml_texto = "No disponible"
        st.warning(
            "El modelo ML fue cargado, pero no pudo realizar la predicción. "
            "Verificá que las columnas del modelo coincidan con lluvia_mm, humedad_suelo, capacidad_drenaje y pendiente_topografica."
        )
else:
    prediccion_ml_texto = "Modelo no encontrado"


# =====================================================
# RIESGO DIFUSO
# =====================================================

riesgo_final, lluvia_fz, humedad_fz, drenaje_fz, pendiente_fz, riesgo_fz = calcular_riesgo_difuso(
    lluvia=lluvia,
    humedad=humedad,
    drenaje=drenaje,
    pendiente=pendiente
)

nivel_riesgo, color_riesgo, clase_riesgo = obtener_nivel_riesgo(riesgo_final)


# =====================================================
# DATAFRAME DE VARIABLES
# =====================================================

df_inputs = pd.DataFrame(
    {
        "Variable": [
            "Lluvia acumulada",
            "Humedad",
            "Capacidad de drenaje",
            "Pendiente topográfica"
        ],
        "Valor": [
            lluvia,
            humedad,
            drenaje,
            pendiente
        ],
        "Máximo de referencia": [
            350,
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
    }
)

df_inputs["Porcentaje"] = (
    df_inputs["Valor"] / df_inputs["Máximo de referencia"] * 100
).clip(upper=100).round(2)

df_inputs["Etiqueta"] = df_inputs["Valor"].round(2).astype(str) + " " + df_inputs["Unidad"]


# =====================================================
# PESTAÑAS PRINCIPALES
# =====================================================

tab_resumen, tab_datos, tab_tecnico = st.tabs(
    [
        "Resumen ejecutivo",
        "Datos climáticos",
        "Detalle técnico del modelo"
    ]
)


# =====================================================
# TAB 1: RESUMEN EJECUTIVO
# =====================================================

with tab_resumen:
    st.markdown('<div class="section-title">Panel general de resultados</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Ciudad analizada</div>
                <div class="card-value">{ciudad}</div>
                <div class="card-small">Ubicación utilizada para la consulta climática y el mapa.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Lluvia acumulada</div>
                <div class="card-value">{lluvia:.2f} mm</div>
                <div class="card-small">Dato climático principal del análisis.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Predicción ML</div>
                <div class="card-value">{prediccion_ml_texto}</div>
                <div class="card-small">Clasificación generada por el modelo entrenado.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Riesgo difuso</div>
                <div class="card-value">{riesgo_final:.2f}/100</div>
                <div class="card-small">Resultado calculado mediante lógica difusa.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="{clase_riesgo}">
        Nivel de riesgo interpretado: {nivel_riesgo}. 
        {obtener_interpretacion(riesgo_final)}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Visualización geográfica y semáforo de riesgo</div>', unsafe_allow_html=True)

    col_mapa, col_gauge = st.columns([1.2, 1])

    with col_mapa:
        mapa = crear_mapa_satelital(
            latitud=latitud,
            longitud=longitud,
            ciudad=ciudad,
            riesgo_final=riesgo_final,
            nivel_riesgo=nivel_riesgo,
            color_riesgo=color_riesgo
        )

        st_folium(
            mapa,
            width=700,
            height=430,
            returned_objects=[]
        )

    with col_gauge:
        fig_gauge = graficar_gauge(
            riesgo_final=riesgo_final,
            nivel_riesgo=nivel_riesgo,
            color_riesgo=color_riesgo
        )

        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown('<div class="section-title">Recomendaciones automáticas</div>', unsafe_allow_html=True)

    recomendaciones = generar_recomendaciones(
        lluvia=lluvia,
        humedad=humedad,
        drenaje=drenaje,
        pendiente=pendiente,
        riesgo_final=riesgo_final
    )

    for recomendacion in recomendaciones:
        st.markdown(
            f"""
            <div class="card">
                <div style="font-size:16px; color:#334155; line-height:1.55;">
                    {recomendacion}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =====================================================
# TAB 2: DATOS CLIMATICOS
# =====================================================

with tab_datos:
    st.markdown('<div class="section-title">Datos climáticos utilizados</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Fuente de datos</div>
            <div style="font-size:17px; color:#334155; line-height:1.65;">
                {fuente_datos}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Tabla de variables utilizadas")
    st.dataframe(
        df_inputs,
        use_container_width=True,
        hide_index=True
    )

    st.plotly_chart(
        graficar_variables(df_inputs),
        use_container_width=True
    )

    if modo_analisis == "Caso histórico Bahía Blanca 2025":
        st.markdown('<div class="section-title">Detalle del caso histórico de Bahía Blanca</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Período analizado</div>
                <div style="font-size:16px; color:#334155; line-height:1.65;">
                    Se analiza el período completo del {CASO_HISTORICO_BAHIA["fecha_inicio"]} al {CASO_HISTORICO_BAHIA["fecha_fin"]}.
                    El valor principal utilizado para el cálculo de riesgo es la lluvia acumulada documentada de
                    {CASO_HISTORICO_BAHIA["lluvia_acumulada_documentada_mm"]} mm.
                    También se considera el evento principal del {CASO_HISTORICO_BAHIA["fecha_evento_principal"]},
                    con {CASO_HISTORICO_BAHIA["lluvia_evento_principal_mm"]} mm en 12 horas.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if df_diario_historico is not None:
            st.markdown("#### Precipitación diaria estimada por Open-Meteo Archive")
            st.plotly_chart(
                graficar_precipitacion_diaria(df_diario_historico),
                use_container_width=True
            )

            st.markdown("#### Días con precipitación dentro del período")
            st.dataframe(
                dias_con_lluvia,
                use_container_width=True,
                hide_index=True
            )

    if df_clima is not None:
        st.markdown('<div class="section-title">Serie climática obtenida desde la API</div>', unsafe_allow_html=True)

        st.plotly_chart(
            graficar_serie_climatica(df_clima),
            use_container_width=True
        )


# =====================================================
# TAB 3: DETALLE TECNICO
# =====================================================

with tab_tecnico:
    st.markdown('<div class="section-title">Lectura del modelo de Machine Learning</div>', unsafe_allow_html=True)

    if modelo is not None:
        modelo_base = modelo

        if hasattr(modelo, "named_steps"):
            modelo_base = list(modelo.named_steps.values())[-1]

        if hasattr(modelo_base, "feature_importances_"):
            importancias = modelo_base.feature_importances_

            nombres_variables = [
                "Lluvia",
                "Humedad",
                "Drenaje",
                "Pendiente"
            ]

            if len(importancias) == len(nombres_variables):
                df_importancia = pd.DataFrame(
                    {
                        "Variable": nombres_variables,
                        "Importancia": importancias
                    }
                ).sort_values("Importancia", ascending=True)

                fig_imp = px.bar(
                    df_importancia,
                    x="Importancia",
                    y="Variable",
                    orientation="h",
                    text=df_importancia["Importancia"].round(3),
                    color="Importancia",
                    color_continuous_scale="Blues"
                )

                fig_imp.update_traces(textposition="outside")

                fig_imp.update_layout(
                    title="Importancia de variables según el modelo entrenado",
                    xaxis_title="Importancia relativa",
                    yaxis_title="",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=380,
                    font=dict(color=COLOR_AZUL),
                    coloraxis_showscale=False
                )

                st.plotly_chart(fig_imp, use_container_width=True)

            else:
                st.info(
                    "El modelo posee importancias, pero la cantidad de variables no coincide con las variables actuales."
                )

        else:
            st.info(
                "El modelo cargado no expone feature_importances_. Esto puede ocurrir si el modelo entrenado "
                "no es un árbol de decisión, random forest u otro modelo basado en importancia de variables."
            )
    else:
        st.info(
            "No se encontró el archivo modelo_ml.pkl. La aplicación continúa funcionando con lógica difusa y visualización."
        )

    st.markdown('<div class="section-title">Funciones de pertenencia de la lógica difusa</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Lluvia",
            "Humedad",
            "Drenaje",
            "Pendiente",
            "Riesgo"
        ]
    )

    with tab1:
        st.plotly_chart(
            graficar_pertenencia(
                lluvia_fz,
                lluvia,
                "Pertenencia difusa de la lluvia",
                "mm"
            ),
            use_container_width=True
        )

    with tab2:
        st.plotly_chart(
            graficar_pertenencia(
                humedad_fz,
                humedad,
                "Pertenencia difusa de la humedad",
                "%"
            ),
            use_container_width=True
        )

    with tab3:
        st.plotly_chart(
            graficar_pertenencia(
                drenaje_fz,
                drenaje,
                "Pertenencia difusa de la capacidad de drenaje",
                "%"
            ),
            use_container_width=True
        )

    with tab4:
        st.plotly_chart(
            graficar_pertenencia(
                pendiente_fz,
                pendiente,
                "Pertenencia difusa de la pendiente topográfica",
                "%"
            ),
            use_container_width=True
        )

    with tab5:
        st.plotly_chart(
            graficar_pertenencia(
                riesgo_fz,
                riesgo_final,
                "Pertenencia difusa del riesgo final",
                "puntos"
            ),
            use_container_width=True
        )

    st.markdown('<div class="section-title">Descripción técnica del sistema</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div style="font-size:16px; color:#334155; line-height:1.65;">
                Esta aplicación integra un modelo de Machine Learning, un sistema de lógica difusa,
                datos climáticos externos y visualización geográfica. La lluvia y la humedad provienen de la API
                o del caso histórico documentado. La capacidad de drenaje y la pendiente son ingresadas manualmente
                porque corresponden a variables territoriales que, en una implementación real, deberían obtenerse
                a partir de cartografía urbana, modelos digitales de elevación, infraestructura pluvial y datos
                geoespaciales oficiales.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
