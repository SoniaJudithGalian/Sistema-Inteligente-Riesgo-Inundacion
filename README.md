# Sistema-Inteligente-Riesgo-Inundacion
Sistema inteligente híbrido para predicción de riesgo de inundaciones utilizando Machine Learning y Lógica Difusa.
# 🌊 Sistema Inteligente Híbrido para Predicción de Riesgo de Inundaciones

Sistema inteligente híbrido para predicción y evaluación de riesgo de inundaciones utilizando Machine Learning y Lógica Difusa.

El proyecto integra modelos predictivos basados en Random Forest con un sistema de inferencia difusa para analizar variables hidrometeorológicas como precipitación, humedad del suelo, capacidad de drenaje y pendiente topográfica.

---

# 🚀 Características

✔ Predicción automática de riesgo de inundación  
✔ Integración de Machine Learning + Lógica Difusa  
✔ Dashboard interactivo con Streamlit  
✔ Visualización de datos y métricas  
✔ Sistema de inferencia difusa  
✔ Clasificación inteligente de riesgo hídrico  
✔ Interpretabilidad del modelo  
✔ Análisis exploratorio de datos (EDA)

---

# 🧠 Tecnologías Utilizadas

- Python
- Scikit-Learn
- Scikit-Fuzzy
- Pandas
- NumPy
- Streamlit
- Matplotlib
- Seaborn

---

# 📊 Variables del Sistema

## Variables de Entrada

| Variable | Descripción |
|---|---|
| lluvia_mm | Precipitación acumulada |
| humedad_suelo | Humedad del suelo |
| capacidad_drenaje | Capacidad de drenaje |
| pendiente_topografica | Pendiente del terreno |

---

## Variable de Salida

| Variable | Descripción |
|---|---|
| riesgo_inundacion | Nivel de riesgo hídrico |

Categorías:
- Bajo
- Medio
- Alto
- Crítico

---

# 🤖 Machine Learning

El modelo de Machine Learning fue entrenado utilizando Random Forest para aprender patrones asociados al riesgo de inundación a partir de datos hidrometeorológicos simulados.

El sistema permite:
- entrenar modelos predictivos
- realizar predicciones automáticas
- evaluar métricas de rendimiento
- identificar variables importantes

---

# 🌫️ Lógica Difusa

El sistema difuso permite representar incertidumbre y escenarios graduales de riesgo utilizando funciones de membresía y reglas difusas.

Ejemplo de regla:

```python
SI lluvia ES alta Y drenaje ES bajo
ENTONCES riesgo ES alto
