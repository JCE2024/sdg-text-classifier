"""Carga del modelo entrenado y generación de predicciones.

Separa la lógica del modelo de la interfaz: `streamlit_app.py` se encarga de la
presentación y delega aquí todo lo que tenga que ver con el modelo.

El notebook deja dos artefactos en `resources/models/`:

* `preparacion.joblib`: el `Pipeline` de scikit-learn con la bolsa de palabras
  TF-IDF y el LSA, junto con la lista de clases en el orden de las neuronas de
  salida de la red.
* `modelo_mlp.keras`: el perceptrón multicapa entrenado.

Un texto nuevo pasa por el mismo pipeline del entrenamiento, así que recibe
exactamente el mismo tratamiento que los textos con los que se ajustó el modelo.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.DataPreprocessing import NOMBRES_ODS

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_MODELOS_POR_DEFECTO = RAIZ_PROYECTO / "resources" / "models"


class ModelController:
    """Envuelve el pipeline de preparación y la red entrenada."""

    def __init__(self, ruta_modelos=None):
        self.ruta_modelos = Path(ruta_modelos) if ruta_modelos else RUTA_MODELOS_POR_DEFECTO
        ruta_preparacion = self.ruta_modelos / "preparacion.joblib"
        ruta_red = self.ruta_modelos / "modelo_mlp.keras"

        faltantes = [r.name for r in (ruta_preparacion, ruta_red) if not r.exists()]
        if faltantes:
            raise FileNotFoundError(
                f"Faltan los artefactos {faltantes} en {self.ruta_modelos}.\n"
                "Ejecuta notebooks/Microproyecto2.ipynb hasta la sección 9 para generarlos."
            )

        artefacto = joblib.load(ruta_preparacion)
        self.preparacion = artefacto["preparacion"]
        self.clases = [int(c) for c in artefacto["clases"]]
        # compile=False porque la aplicación solo predice, nunca reentrena
        self.modelo = load_model(ruta_red, compile=False)

    def probabilidades(self, textos):
        """Devuelve la distribución de probabilidad sobre los ODS de cada texto."""
        x = self.preparacion.transform(list(textos))
        return self.modelo.predict(x, verbose=0)

    def predecir(self, texto, top_n=3):
        """Predice el ODS de un texto libre.

        Devuelve el ODS ganador y las `top_n` alternativas más probables. Mostrar
        las alternativas es útil en la interfaz porque varios ODS se solapan
        temáticamente, por ejemplo el 1 y el 10, o el 13 y el 15.
        """
        if not texto or not texto.strip():
            raise ValueError("El texto de entrada está vacío.")

        distribucion = self.probabilidades([texto])[0]
        orden = np.argsort(distribucion)[::-1][:top_n]
        ganador = self.clases[int(np.argmax(distribucion))]

        return {
            "ods": ganador,
            "nombre": NOMBRES_ODS.get(ganador, "Desconocido"),
            "confianza": float(distribucion.max()),
            "alternativas": [
                {
                    "ods": self.clases[i],
                    "nombre": NOMBRES_ODS.get(self.clases[i], "Desconocido"),
                    "probabilidad": float(distribucion[i]),
                }
                for i in orden
            ],
        }

    def predecir_lote(self, textos):
        """Predice sobre una colección de textos y devuelve un DataFrame."""
        textos = list(textos)
        distribuciones = self.probabilidades(textos)
        predicciones = [self.clases[i] for i in distribuciones.argmax(axis=1)]

        return pd.DataFrame({
            "texto": [t[:200] + "..." if len(str(t)) > 200 else t for t in textos],
            "ods_predicho": predicciones,
            "nombre_ods": [NOMBRES_ODS.get(p, "Desconocido") for p in predicciones],
            "confianza": distribuciones.max(axis=1),
        })
