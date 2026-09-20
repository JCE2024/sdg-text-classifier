"""Preparación de textos en español para la clasificación por ODS.

Este módulo concentra TODA la lógica de preparación de datos por una razón
práctica: el notebook entrena y guarda el pipeline con `joblib`, y la app de
Streamlit lo vuelve a cargar. Al deserializar, joblib necesita volver a
importar las clases desde el mismo módulo y con el mismo nombre. Si el
transformador se definiera dentro del notebook, la app fallaría al cargar el
artefacto con un error del tipo `AttributeError: Can't get attribute ...`.

Regla práctica: cualquier clase que termine dentro de un archivo .joblib debe
estar definida aquí, nunca en una celda del notebook.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

SEED = 32

# Nombres oficiales de los 17 ODS. El conjunto de datos de este microproyecto
# solo contiene las clases 1 a 16 (el ODS 17 no aparece), por lo que las
# etiquetas de los reportes se deben derivar de los datos y no de este
# diccionario.
NOMBRES_ODS = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
    17: "Alianzas para lograr los objetivos",
}

# Lista de respaldo por si `nltk_data` no está disponible. Es el caso típico de
# Streamlit Community Cloud, donde no conviene descargar corpus en cada arranque.
_STOPWORDS_RESPALDO = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
    "contra", "cual", "cuando", "de", "del", "desde", "donde", "durante", "e",
    "el", "ella", "ellas", "ellos", "en", "entre", "era", "eran", "eres", "es",
    "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estado", "están",
    "estar", "estas", "este", "esto", "estos", "ha", "había", "han", "hasta",
    "hay", "la", "las", "le", "les", "lo", "los", "más", "me", "mi", "mis",
    "mucho", "muy", "nada", "ni", "no", "nos", "nosotros", "o", "otra",
    "otras", "otro", "otros", "para", "pero", "poco", "por", "porque", "que",
    "quien", "se", "sea", "ser", "si", "sin", "sobre", "son", "su", "sus",
    "también", "tanto", "te", "tiene", "tienen", "todo", "todos", "tu", "un",
    "una", "uno", "unos", "y", "ya", "yo",
}

# Acepta letras acentuadas para el caso en que se decida NO eliminar acentos.
_PATRON_TOKEN = re.compile(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]{2,}")


@lru_cache(maxsize=1)
def cargar_stopwords() -> frozenset:
    """Stopwords en español, con respaldo si `nltk_data` no está descargado."""
    try:
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("spanish"))
    except Exception:  # LookupError si falta el corpus, ImportError si falta nltk
        return frozenset(_STOPWORDS_RESPALDO)


@lru_cache(maxsize=1)
def cargar_stemmer():
    """SnowballStemmer en español (el PorterStemmer de los tutoriales es de inglés)."""
    from nltk.stem import SnowballStemmer

    return SnowballStemmer("spanish")


def quitar_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)
    )


class LimpiadorTextoEspanol(BaseEstimator, TransformerMixin):
    """Normaliza, tokeniza, filtra stopwords y opcionalmente aplica stemming.

    Recibe un iterable de textos y devuelve una lista de cadenas ya limpias,
    lista para el `TfidfVectorizer` que viene a continuación en el pipeline.

    Los hiperparámetros son banderas simples a propósito, para poder explorarlos
    con `GridSearchCV` sobre el pipeline completo, por ejemplo con
    `{"preparacion__limpieza__aplicar_stemming": [True, False]}`.
    """

    def __init__(
        self,
        eliminar_acentos: bool = True,
        aplicar_stemming: bool = True,
        longitud_minima: int = 3,
    ) -> None:
        self.eliminar_acentos = eliminar_acentos
        self.aplicar_stemming = aplicar_stemming
        self.longitud_minima = longitud_minima

    def fit(self, X, y=None):  # noqa: N803 - convención de scikit-learn
        return self

    def transform(self, X):  # noqa: N803
        stopwords_es = cargar_stopwords()
        stemmer = cargar_stemmer() if self.aplicar_stemming else None

        limpios = []
        for texto in X:
            texto = str(texto).lower()
            if self.eliminar_acentos:
                texto = quitar_acentos(texto)
            tokens = _PATRON_TOKEN.findall(texto)
            tokens = [
                t
                for t in tokens
                if t not in stopwords_es and len(t) >= self.longitud_minima
            ]
            if stemmer is not None:
                tokens = [stemmer.stem(t) for t in tokens]
            limpios.append(" ".join(tokens))
        return limpios


def construir_pipeline_preparacion(
    n_componentes: int = 15,
    max_features: int | None = 20000,
    min_df: int = 5,
    max_df: float = 0.8,
    ngram_range: tuple = (1, 1),
    aplicar_stemming: bool = True,
    semilla: int = SEED,
) -> Pipeline:
    """Pipeline de preparación: limpieza -> TF-IDF -> LSA (TruncatedSVD).

    `TruncatedSVD` se aplica directamente sobre la matriz dispersa TF-IDF sin
    centrar los datos, que es justamente lo que define el análisis semántico
    latente (LSA). Usar PCA obligaría a densificar la matriz.
    """
    return Pipeline(
        steps=[
            ("limpieza", LimpiadorTextoEspanol(aplicar_stemming=aplicar_stemming)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    max_df=max_df,
                    ngram_range=ngram_range,
                    sublinear_tf=True,
                ),
            ),
            ("lsa", TruncatedSVD(n_components=n_componentes, random_state=semilla)),
        ]
    )


def cargar_datos(ruta: str = "data/Datos_textosODS.xlsx") -> pd.DataFrame:
    """Carga el conjunto de datos y valida que tenga las columnas esperadas."""
    df = pd.read_excel(ruta)
    faltantes = {"textos", "ODS"} - set(df.columns)
    if faltantes:
        raise ValueError(f"El archivo no tiene las columnas esperadas: {faltantes}")
    return df


def terminos_por_topico(pipeline: Pipeline, n_terminos: int = 12) -> pd.DataFrame:
    """Palabras de mayor peso en cada componente del LSA, para interpretar tópicos.

    Cada fila es una componente del SVD. Una componente es una dirección en el
    espacio de términos, y sus pesos más altos son las palabras que la definen.
    """
    vocabulario = pipeline.named_steps["tfidf"].get_feature_names_out()
    lsa = pipeline.named_steps["lsa"]

    filas = []
    for i, componente in enumerate(lsa.components_):
        indices = componente.argsort()[::-1][:n_terminos]
        filas.append(
            {
                "topico": f"Tópico {i + 1}",
                "varianza_explicada": lsa.explained_variance_ratio_[i],
                "terminos": ", ".join(vocabulario[j] for j in indices),
            }
        )
    return pd.DataFrame(filas)


def etiquetas_ods(clases) -> list:
    """Convierte códigos de ODS en etiquetas legibles para los reportes."""
    return [f"ODS {int(c)} - {NOMBRES_ODS.get(int(c), 'Desconocido')}" for c in clases]
