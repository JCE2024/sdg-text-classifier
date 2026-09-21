"""Procesamiento de los textos en español para la clasificación por ODS.

La función `procesar_texto()` está definida aquí, y no solo en el notebook, porque
joblib guarda una referencia al módulo donde vive la función, no su código. La
aplicación de Streamlit la vuelve a importar desde este archivo cuando carga el
pipeline de preparación. Si únicamente existiera en una celda del notebook, la
carga fallaría con `AttributeError: Can't get attribute 'procesar_texto'`.
"""

import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

SEED = 32

# Nombres oficiales de los 17 ODS. El conjunto de datos del microproyecto solo
# contiene las clases 1 a 16, así que las etiquetas de los reportes se derivan de
# los datos y no de este diccionario.
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

TOKENIZADOR = RegexpTokenizer(r"\w+")
STEMMER = SnowballStemmer("spanish")

# En Streamlit Community Cloud el corpus de nltk no viene descargado.
try:
    PALABRAS_VACIAS = set(stopwords.words("spanish"))
except LookupError:
    nltk.download("stopwords")
    PALABRAS_VACIAS = set(stopwords.words("spanish"))


def procesar_texto(texto):
    """Pasa a minúsculas, tokeniza, quita palabras vacías y aplica stemming."""
    tokens = TOKENIZADOR.tokenize(str(texto).lower())
    tokens = [palabra for palabra in tokens if palabra not in PALABRAS_VACIAS]
    tokens = [STEMMER.stem(palabra) for palabra in tokens]
    return " ".join(tokens)
