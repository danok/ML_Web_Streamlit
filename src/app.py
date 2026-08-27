# from utils import db_connect
# engine = db_connect()
# your code here
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Clasificador de billetes",
    page_icon="💵",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "banknote_model.pkl"

@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)

modelo = cargar_modelo()

st.title("💵 Clasificador de billetes")
st.write(
    "Introduce las características del billete para determinar "
    "si el modelo lo clasifica como auténtico o falso."
)

variance = st.number_input("Variance", value=0.0, format="%.5f")
skewness = st.number_input("Skewness", value=0.0, format="%.5f")
curtosis = st.number_input("Curtosis", value=0.0, format="%.5f")
entropy = st.number_input("Entropy", value=0.0, format="%.5f")

if st.button("Clasificar"):

    datos = pd.DataFrame(
        [[variance, skewness, curtosis, entropy]],
        columns=[
            "variance",
            "skewness",
            "curtosis",
            "entropy"
        ]
    )

    prediccion = int(modelo.predict(datos)[0])

    if prediccion == 0:
        st.success("Resultado: Billete auténtico")
    else:
        st.error("Resultado: Billete falso")

    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(datos)[0]

        st.subheader("Probabilidades")

        st.write(
            f"Auténtico: {probabilidades[0]:.2%}"
        )

        st.write(
            f"Falso: {probabilidades[1]:.2%}"
        )