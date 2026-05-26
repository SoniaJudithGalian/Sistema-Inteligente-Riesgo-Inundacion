# 🌊 Sistema Inteligente de Riesgo de Inundaciones

Este proyecto consiste en el desarrollo de un sistema inteligente para evaluar el riesgo de inundaciones utilizando Machine Learning y Lógica Difusa.

## 🎯 Objetivo

El objetivo principal es construir una herramienta capaz de asistir en la toma de decisiones preventivas frente a posibles inundaciones, considerando variables climáticas e hidrológicas.

## 🧠 Tecnologías utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Scikit-fuzzy
- Matplotlib
- Plotly
- Joblib
- Google Colab
- GitHub

## 📊 Variables del sistema

El sistema utiliza las siguientes variables de entrada:

- Lluvia acumulada en milímetros
- Humedad del suelo
- Capacidad de drenaje
- Pendiente topográfica

## 🤖 Machine Learning

Se entrenó un modelo de clasificación Random Forest para predecir el nivel de riesgo de inundación.

Las clases utilizadas fueron:

- 🟢 Bajo
- 🟡 Medio
- 🟠 Alto
- 🔴 Crítico

El modelo fue entrenado en Google Colab con datos simulados y luego exportado como `modelo_ml.pkl`.

## 🌫️ Lógica Difusa

Además del modelo de Machine Learning, se implementó un sistema experto difuso.

Este sistema utiliza reglas del tipo:

- Si la lluvia es alta y el drenaje es bajo, entonces el riesgo es alto.
- Si la lluvia es baja y el drenaje es alto, entonces el riesgo es bajo.
- Si la humedad está saturada y la lluvia es alta, entonces el riesgo aumenta.

La lógica difusa permite interpretar situaciones inciertas y graduales.

## 🖥️ Aplicación Streamlit

La aplicación permite al usuario ingresar valores mediante sliders:

- Lluvia
- Humedad
- Drenaje
- Pendiente

Luego muestra:

- Predicción del modelo ML
- Resultado del sistema difuso
- Alertas visuales
- Gráficos de variables
- Importancia de variables
- Medidor de riesgo
- Interpretación final automática

## 📁 Estructura del proyecto

```bash
sistema-riesgo-inundaciones/
│
├── app.py
├── modelo_ml.pkl
├── README.md
├── requirements.txt
└── notebooks/
    ├── entrenamiento_modelo_ml.ipynb
    └── sistema_experto_difuso.ipynb

