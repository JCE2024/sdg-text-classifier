"""Aplicación Streamlit para el bono del microproyecto 2.

Cumple los tres requisitos del bono: recibe texto libre, lo procesa con el
MISMO pipeline construido en el notebook (cargado desde el artefacto .joblib)
y devuelve el ODS predicho.

Ejecución local:
    streamlit run streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.DataPreprocessing import NOMBRES_ODS
from src.ModelController import ModelController

st.set_page_config(page_title="Clasificador de textos por ODS", page_icon="🌍", layout="wide")

EJEMPLOS = {
    "Educación": (
        "Es necesario garantizar que todas las niñas y niños terminen la enseñanza "
        "primaria y secundaria de calidad, con docentes cualificados y materiales "
        "pedagógicos pertinentes para su contexto."
    ),
    "Agua": (
        "El acceso al agua potable y al saneamiento básico sigue siendo limitado en "
        "las zonas rurales, lo que aumenta la incidencia de enfermedades y afecta "
        "especialmente a los hogares más pobres."
    ),
    "Energía": (
        "La transición hacia fuentes de energía renovable, como la solar y la eólica, "
        "permite reducir la dependencia de los combustibles fósiles y ampliar la "
        "cobertura eléctrica en comunidades aisladas."
    ),
}


@st.cache_resource
def cargar_controlador() -> ModelController:
    """Carga el modelo una sola vez y lo reutiliza entre interacciones."""
    return ModelController()


def main() -> None:
    st.title("🌍 Clasificación de textos según los ODS")
    st.caption(
        "Microproyecto 2 — Machine Learning No Supervisado (MAIA). "
        "El modelo asigna a un texto en español el Objetivo de Desarrollo "
        "Sostenible con el que guarda mayor relación semántica."
    )

    try:
        controlador = cargar_controlador()
    except FileNotFoundError as error:
        st.error(str(error))
        st.info(
            "El archivo `resources/models/modelo_ods.joblib` no está versionado "
            "hasta que entrenes el modelo. Ejecuta el notebook y vuelve a desplegar."
        )
        st.stop()

    tab_texto, tab_archivo = st.tabs(["Texto libre", "Archivo CSV / Excel"])

    with tab_texto:
        ejemplo = st.selectbox(
            "Cargar un ejemplo (opcional)", ["—"] + list(EJEMPLOS), index=0
        )
        texto = st.text_area(
            "Escribe o pega un texto para clasificar",
            value=EJEMPLOS.get(ejemplo, ""),
            height=200,
            placeholder="Por ejemplo: un fragmento de un plan de desarrollo territorial...",
        )

        if st.button("Clasificar", type="primary"):
            if not texto.strip():
                st.warning("Ingresa un texto antes de clasificar.")
            else:
                resultado = controlador.predecir(texto)
                st.success(
                    f"**ODS {resultado['ods']} — {resultado['nombre']}**"
                )
                if resultado["confianza"] is not None:
                    st.metric("Confianza del modelo", f"{resultado['confianza']:.1%}")
                    st.subheader("Alternativas más probables")
                    alternativas = pd.DataFrame(resultado["alternativas"])
                    alternativas["probabilidad"] = alternativas["probabilidad"].map("{:.1%}".format)
                    st.dataframe(alternativas, hide_index=True, use_container_width=True)
                    st.caption(
                        "Varios ODS se solapan temáticamente, por lo que conviene "
                        "revisar las alternativas y no solo la clase ganadora."
                    )

    with tab_archivo:
        st.write(
            "Carga un archivo con una columna llamada `textos` para clasificar "
            "varios documentos a la vez."
        )
        archivo = st.file_uploader("Archivo", type=["csv", "xlsx"])
        if archivo is not None:
            df = (
                pd.read_csv(archivo)
                if Path(archivo.name).suffix == ".csv"
                else pd.read_excel(archivo)
            )
            if "textos" not in df.columns:
                st.error(
                    f"El archivo debe tener una columna `textos`. "
                    f"Columnas encontradas: {list(df.columns)}"
                )
            else:
                resultados = controlador.predecir_lote(df["textos"])
                st.dataframe(resultados, use_container_width=True)
                st.bar_chart(resultados["ods_predicho"].value_counts().sort_index())

    with st.sidebar:
        st.header("Sobre el modelo")
        st.write(
            "El texto pasa por el mismo pipeline del entrenamiento: "
            "limpieza en español (stopwords y stemming), vectorización TF-IDF "
            "y reducción de dimensionalidad con LSA (`TruncatedSVD`)."
        )
        st.write(
            f"Clases disponibles: **{len(controlador.clases)}** "
            f"(ODS {min(controlador.clases)} a {max(controlador.clases)})."
        )
        with st.expander("Ver los 17 ODS"):
            for codigo, nombre in NOMBRES_ODS.items():
                st.write(f"**{codigo}.** {nombre}")


if __name__ == "__main__":
    main()
