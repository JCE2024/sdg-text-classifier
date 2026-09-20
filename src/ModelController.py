"""Carga del modelo entrenado y generación de predicciones.

Separa la lógica del modelo de la interfaz: `streamlit_app.py` solo se encarga
de la presentación y delega aquí todo lo que tenga que ver con el modelo.

El artefacto esperado es un `Pipeline` completo de scikit-learn que incluye la
preparación (limpieza -> TF-IDF -> LSA) y el clasificador final. Guardar un
único objeto evita el error clásico del despliegue: aplicar al texto del
usuario un preprocesamiento distinto del usado en entrenamiento.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.DataPreprocessing import NOMBRES_ODS

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_MODELO_POR_DEFECTO = RAIZ_PROYECTO / "resources" / "models" / "modelo_ods.joblib"


class ModelController:
    """Envuelve el pipeline entrenado y expone predicciones legibles."""

    def __init__(self, ruta_modelo: str | Path | None = None) -> None:
        self.ruta_modelo = Path(ruta_modelo) if ruta_modelo else RUTA_MODELO_POR_DEFECTO
        if not self.ruta_modelo.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {self.ruta_modelo}.\n"
                "Ejecuta el notebook notebooks/Microproyecto2.ipynb hasta la sección "
                "de serialización para generar el artefacto."
            )
        self.pipeline = joblib.load(self.ruta_modelo)
        self.clases = list(self.pipeline.classes_)

    @property
    def soporta_probabilidades(self) -> bool:
        return hasattr(self.pipeline, "predict_proba")

    def predecir(self, texto: str, top_n: int = 3) -> dict:
        """Predice el ODS de un texto libre.

        Devuelve el ODS ganador y, si el clasificador expone probabilidades,
        las `top_n` alternativas más probables. Mostrar las alternativas es útil
        en la interfaz porque varios ODS se solapan temáticamente (por ejemplo
        el 1 y el 10, o el 13 y el 15).
        """
        if not texto or not texto.strip():
            raise ValueError("El texto de entrada está vacío.")

        # El pipeline espera un iterable de documentos, no una cadena suelta.
        prediccion = int(self.pipeline.predict([texto])[0])
        resultado = {
            "ods": prediccion,
            "nombre": NOMBRES_ODS.get(prediccion, "Desconocido"),
            "confianza": None,
            "alternativas": [],
        }

        if self.soporta_probabilidades:
            probabilidades = self.pipeline.predict_proba([texto])[0]
            orden = np.argsort(probabilidades)[::-1][:top_n]
            resultado["confianza"] = float(probabilidades[list(self.clases).index(prediccion)])
            resultado["alternativas"] = [
                {
                    "ods": int(self.clases[i]),
                    "nombre": NOMBRES_ODS.get(int(self.clases[i]), "Desconocido"),
                    "probabilidad": float(probabilidades[i]),
                }
                for i in orden
            ]
        return resultado

    def predecir_lote(self, textos) -> pd.DataFrame:
        """Predice sobre una colección de textos y devuelve un DataFrame."""
        textos = list(textos)
        predicciones = self.pipeline.predict(textos)
        datos = {
            "texto": [t[:200] + "..." if len(str(t)) > 200 else t for t in textos],
            "ods_predicho": [int(p) for p in predicciones],
            "nombre_ods": [NOMBRES_ODS.get(int(p), "Desconocido") for p in predicciones],
        }
        if self.soporta_probabilidades:
            probabilidades = self.pipeline.predict_proba(textos)
            datos["confianza"] = probabilidades.max(axis=1)
        return pd.DataFrame(datos)
