# Microproyecto 2 — Clasificación de textos según los ODS

Solución al microproyecto 2 de **Machine Learning No Supervisado** (Maestría en Inteligencia
Artificial, Universidad de los Andes).

El proyecto construye una solución de procesamiento de lenguaje natural que asigna a un texto
en español el **Objetivo de Desarrollo Sostenible (ODS)** con el que guarda mayor relación
semántica, con el fin de apoyar el análisis de información textual en procesos de planeación
participativa territorial.

El método combina una representación de **bolsa de palabras con pesado TF-IDF**, un modelo de
tópicos por **análisis semántico latente (LSA)** con `TruncatedSVD`, y un clasificador
supervisado sobre el espacio reducido. Todo ello encadenado en un `Pipeline` de scikit-learn.

## Estructura

```
Microproyecto2/
├── data/
│   └── Datos_textosODS.xlsx        Conjunto de datos (9.656 textos etiquetados)
├── docs/                           (solo local, no versionado)
│   ├── Microproyecto2.pdf          Enunciado y rúbrica de evaluación
│   ├── BonoOpcionalMicroproyecto2.png
│   └── PCA_Despliegue_guia_streamlit.ipynb   Guía del curso para el despliegue
├── notebooks/
│   ├── Microproyecto2.ipynb        ← el entregable principal
│   └── referencia/                 (solo local) Tutoriales del curso: MLP y autoencoders
├── src/
│   ├── DataPreprocessing.py        Limpieza en español, pipeline TF-IDF + LSA, utilidades
│   └── ModelController.py          Carga del modelo y predicción
├── resources/models/               Aquí se guarda modelo_ods.joblib al entrenar
├── streamlit_app.py                Aplicación del bono opcional
├── requirements.txt                Dependencias de la aplicación
└── requirements-dev.txt            Dependencias del notebook
```

> `docs/` y `notebooks/referencia/` contienen material del curso y están excluidos del
> control de versiones: existen en la copia local pero no se publican en este repositorio.

## Puesta en marcha

Este proyecto se desarrolló con el intérprete de Anaconda, que ya trae todas las dependencias:

```powershell
& "C:\Users\Usuario\anaconda3\python.exe" -m jupyter lab
```

En un entorno limpio:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt
jupyter lab
```

Abre `notebooks/Microproyecto2.ipynb` y ejecuta las celdas en orden. El notebook agrega
automáticamente la raíz del proyecto al `sys.path`, así que `from src...` funciona sin
configuración adicional.

Exportar el entregable en HTML:

```bash
jupyter nbconvert --to html notebooks/Microproyecto2.ipynb
```

## El conjunto de datos

`data/Datos_textosODS.xlsx` proviene del *OSDG Community Dataset* (versión 2023), traducido al
español con DeepL y aumentado con la API de ChatGPT. Una sola hoja, **9.656 filas**, dos
columnas: `textos` (español, ~709 caracteres en promedio) y `ODS` (entero).

Dos particularidades que condicionan el modelado:

- **Hay 16 clases, no 17.** El enunciado menciona los 17 ODS, pero el ODS 17 (*Alianzas para
  lograr los objetivos*) no aparece en los datos. Las etiquetas de los reportes se derivan de
  los datos; fijarlas en 17 rompe `classification_report`.
- **Las clases están desbalanceadas**: de 312 textos (ODS 12) a 1.080 (ODS 16). De ahí que las
  particiones sean estratificadas y la métrica principal sea el **F1 macro** en lugar de la
  exactitud.

Los textos están en español, por lo que se usan `stopwords` en español y `SnowballStemmer`,
no el `PorterStemmer` de los tutoriales del curso, que es de inglés.

## Correspondencia con la rúbrica

| Sección del notebook | Actividad evaluada | Peso |
|---|---|---|
| 2 y 3 | Preparación de los datos y reducción de dimensionalidad, justificadas | 30% |
| 4 | Construcción del pipeline de preparación | 15% |
| 5 | Modelo LSA e interpretación de al menos 5 tópicos frente a los ODS | 15% |
| 6 | Clasificador con búsqueda de hiperparámetros y métricas justificadas | 30% |
| 7 | Clasificación de al menos 4 textos no usados en el aprendizaje | 10% |

Los bloques marcados con **✍️** dentro del notebook son los que la rúbrica califica: piden
argumentación escrita, no solo resultados numéricos.

## Bono opcional: aplicación en Streamlit

```bash
streamlit run streamlit_app.py
```

La aplicación recibe texto libre, lo procesa con el **mismo** pipeline del entrenamiento y
devuelve el ODS predicho junto con las alternativas más probables. Requiere que
`resources/models/modelo_ods.joblib` exista, lo cual ocurre al ejecutar la sección 8 del
notebook.

### Por qué el preprocesamiento vive en `src/` y no en el notebook

`joblib` no guarda el código de las clases, solo una referencia a su módulo. Al cargar el
modelo, vuelve a importar `LimpiadorTextoEspanol` desde `src.DataPreprocessing`. Si esa clase
estuviera definida en una celda del notebook, la aplicación fallaría con
`AttributeError: Can't get attribute 'LimpiadorTextoEspanol'`. Esta es la causa más común de
que un despliegue funcione en local y se rompa en la nube.

Por el mismo motivo se serializa **un único artefacto** con preparación y clasificador juntos,
en lugar de los tres archivos separados (`model` / `pca` / `scaler`) de la plantilla del curso:
así es imposible que el texto del usuario reciba un tratamiento distinto al del entrenamiento.

### Despliegue en Streamlit Community Cloud

1. El repositorio debe ser **público** y contener el archivo `.joblib` del modelo (la
   aplicación no arranca sin él; por eso no está en `.gitignore`).
2. En *Create app* → *Deploy a public app from GitHub*, indicar `streamlit_app.py` como
   archivo principal.
3. En *Advanced settings*, elegir la versión de Python.

> **Cuidado con la versión de Python.** La guía del curso sugiere fijar Python 3.10, pero este
> proyecto se entrenó con `numpy==2.3.5`, que exige Python ≥ 3.11. Con 3.10 la instalación de
> dependencias falla. Usa **3.11 o superior** (idealmente 3.13, la del entrenamiento), o baja
> `numpy` a una versión compatible con 3.10 y **vuelve a entrenar y serializar el modelo** con
> esa versión.

## Entregables

- [ ] `Microproyecto2.ipynb` con todas las celdas ejecutadas y su salida visible.
- [ ] `Microproyecto2.html` exportado del notebook.
- [ ] (Bono) URL de la aplicación desplegada y URL de este repositorio.
