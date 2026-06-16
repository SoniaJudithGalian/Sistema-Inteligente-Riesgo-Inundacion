#  Sistema Inteligente de Riesgo de Inundaciones

Aplicación interactiva desarrollada con **Python, Streamlit, Machine Learning, Lógica Difusa y API climática**, orientada a estimar el riesgo de inundación a partir de variables climáticas y territoriales.

El sistema permite seleccionar una ciudad, obtener datos climáticos automáticos mediante una API, ingresar variables locales como drenaje y pendiente, y visualizar el nivel de riesgo mediante indicadores, gráficos y un semáforo de alerta.

---

## Ver aplicación en línea

Podés probar la aplicación interactiva desde el siguiente enlace:

🔗 [Ver aplicación en Streamlit](sistema-inteligente-riesgo-inundacion-aozn2fuz2v2a9yd6bewkws)
##  Objetivo del proyecto

El objetivo de este proyecto es construir un sistema inteligente capaz de estimar el riesgo de inundación combinando:

* Un modelo de **Machine Learning** entrenado con datos simulados.
* Un sistema de **Lógica Difusa** para interpretar escenarios graduales e inciertos.
* Una **API climática** para obtener datos automáticos de lluvia y humedad.
* Una interfaz visual e interactiva desarrollada en **Streamlit**.

Este proyecto fue pensado como una solución académica escalable, con potencial aplicación en contextos municipales, urbanos o territoriales.

##  Descripción general

El sistema analiza cuatro variables principales:

| Variable              | Descripción                           | Fuente                         |
| --------------------- | ------------------------------------- | ------------------------------ |
| Lluvia                | Precipitación estimada en mm          | API climática o ingreso manual |
| Humedad del suelo     | Humedad relativa promedio             | API climática o ingreso manual |
| Capacidad de drenaje  | Nivel de drenaje urbano o territorial | Ingreso manual                 |
| Pendiente topográfica | Inclinación del terreno               | Ingreso manual                 |

A partir de estas variables, el sistema genera:

* Predicción del modelo de Machine Learning.
* Valor de riesgo calculado mediante lógica difusa.
* Nivel interpretado del riesgo.
* Visualizaciones interactivas.
* Semáforo de riesgo.
* Gráficos climáticos de precipitación y humedad.

---

##  Funcionamiento del sistema

El flujo general del sistema es el siguiente:

```text
Ciudad seleccionada
        ↓
API climática Open-Meteo
        ↓
Lluvia y humedad estimadas
        ↓
Drenaje y pendiente ingresados manualmente
        ↓
Modelo Machine Learning + Sistema de Lógica Difusa
        ↓
Nivel de riesgo de inundación
        ↓
Visualizaciones e interpretación
```

---

##  Machine Learning

El sistema utiliza un modelo de clasificación entrenado previamente y guardado en formato `.pkl`.

El modelo recibe como entrada:

```python
lluvia_mm
humedad_suelo
capacidad_drenaje
pendiente_topografica
```

Y devuelve una clase de riesgo:

| Clase | Nivel   |
| ----- | ------- |
| 0     | Bajo    |
| 1     | Medio   |
| 2     | Alto    |
| 3     | Crítico |

El modelo fue integrado en la aplicación mediante `joblib`.

---

##  Lógica Difusa

Además del modelo de Machine Learning, el sistema incorpora lógica difusa para interpretar escenarios donde el riesgo no es completamente binario o exacto.

Se definieron funciones de pertenencia para:

* Lluvia baja, media y alta.
* Humedad seca, media y saturada.
* Drenaje bajo, medio y alto.
* Pendiente llana, moderada y empinada.
* Riesgo bajo, medio y alto.

La lógica difusa permite representar mejor situaciones graduales, por ejemplo:

> Una lluvia moderada con bajo drenaje puede generar un riesgo mayor que una lluvia moderada con buen drenaje.

---

## 🚦 Semáforo de riesgo

La aplicación incluye un semáforo visual para facilitar la interpretación del riesgo calculado.

| Color       | Nivel de riesgo |
| ----------- | --------------- |
| 🟢 Verde    | Bajo            |
| 🟡 Amarillo | Medio           |
| 🟠 Naranja  | Alto            |
| 🔴 Rojo     | Crítico         |

Esta visualización permite que el resultado sea comprensible tanto para usuarios técnicos como no técnicos.

---

##  Integración con API climática

El sistema fue escalado incorporando una API climática para obtener datos automáticos de lluvia y humedad.

Actualmente se utiliza la API de **Open-Meteo**, que permite consultar información meteorológica según coordenadas geográficas.

Ciudades incluidas:

* Neuquén Capital
* Añelo
* Rincón de los Sauces
* Centenario
* Plottier
* Cutral Có

La lluvia y la humedad pueden cargarse automáticamente desde la API, mientras que drenaje y pendiente se mantienen como variables configurables por el usuario.

---

##  Visualizaciones incluidas

La aplicación contiene diferentes visualizaciones para mejorar la interpretación del sistema:

* Tarjetas con resultados principales.
* Gráfico de variables ingresadas.
* Semáforo de riesgo difuso.
* Gráficos de precipitación estimada por hora.
* Gráfico de humedad relativa por hora.
* Interpretación automática del resultado.

---

##  Tecnologías utilizadas

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Scikit-fuzzy
* NetworkX
* SciPy
* Plotly
* Matplotlib
* Seaborn
* Requests
* Open-Meteo API

---

##  Estructura del proyecto

```text
sistema-inteligente-riesgo-inundacion/
│
├── app_api_clima.py              # Aplicación principal con API climática
├── app.py                        # Versión base de la aplicación
├── modelo_ml.pkl                 # Modelo de Machine Learning entrenado
├── dataset_riesgo_inundacion.csv # Dataset utilizado para el entrenamiento
├── requirements.txt              # Librerías necesarias
└── README.md                     # Documentación del proyecto
```

---

## ⚙️ Instalación y ejecución local

Para ejecutar el proyecto de forma local:

### 1. Clonar el repositorio

```bash
git clone https://github.com/soniajudithgalian/sistema-inteligente-riesgo-inundacion.git
```

### 2. Entrar a la carpeta del proyecto

```bash
cd sistema-inteligente-riesgo-inundacion
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
streamlit run app_api_clima.py
```

---

##  Dependencias principales

El archivo `requirements.txt` incluye las siguientes versiones:

```text
streamlit==1.39.0
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.6.1
joblib==1.4.2
scikit-fuzzy==0.5.0
networkx==3.2.1
scipy==1.11.4
matplotlib==3.8.4
seaborn==0.13.2
plotly==5.24.1
requests==2.32.3
```

---

##  Estado del proyecto

Proyecto funcional desplegado en Streamlit Cloud.

Actualmente permite:

* Seleccionar ciudad.
* Consultar datos climáticos automáticos.
* Ingresar variables territoriales.
* Calcular riesgo mediante Machine Learning.
* Calcular riesgo mediante lógica difusa.
* Visualizar resultados de forma interactiva.

---

## 🚀 Posibles mejoras futuras

Algunas mejoras posibles para próximas versiones:

* Incorporar mapas interactivos.
* Agregar datos reales de infraestructura de drenaje.
* Integrar información topográfica mediante datos GIS.
* Conectar sensores IoT o estaciones meteorológicas.
* Guardar registros históricos de consultas.
* Agregar alertas automáticas por nivel de riesgo.
* Incorporar más ciudades o zonas rurales.
* Mejorar el modelo con datos reales municipales.

---

##  Conclusión

Este proyecto demuestra cómo se pueden combinar técnicas de Inteligencia Artificial, Machine Learning, lógica difusa y datos climáticos externos para construir una herramienta interactiva de apoyo a la toma de decisiones.

La integración con una API climática permite que el sistema evolucione desde una simulación manual hacia una solución semiautomática y escalable, aplicable a problemáticas territoriales reales como el monitoreo preventivo del riesgo de inundación.

---

## 👩‍💻 Autor

**Grupo 16**

Estudiante de Ciencia de Datos e Inteligencia Artificial.

Proyecto desarrollado como parte de una propuesta académica orientada a soluciones inteligentes para problemáticas territoriales.
