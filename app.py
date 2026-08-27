import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Detector de billetes", page_icon="💵", layout="centered")

@st.cache_resource
def load_model():
    base_dir = Path(__file__).resolve().parent
    return joblib.load(base_dir / "models" / "banknote_model.pkl")

model = load_model()

st.title("💵 Clasificador de billetes")
st.write("Introduce las características del billete para determinar si el modelo lo clasifica como auténtico o falso.")

variance = st.number_input("Variance", value=0.0, format="%.5f")
skewness = st.number_input("Skewness", value=0.0, format="%.5f")
curtosis = st.number_input("Curtosis", value=0.0, format="%.5f")
entropy = st.number_input("Entropy", value=0.0, format="%.5f")

if st.button("Clasificar"):
    datos = pd.DataFrame(
        [[variance, skewness, curtosis, entropy]],
        columns=["variance", "skewness", "curtosis", "entropy"]
    )

    pred = int(model.predict(datos)[0])
    proba = model.predict_proba(datos)[0]

    if pred == 0:
        st.success("Resultado: Billete auténtico")
    else:
        st.error("Resultado: Billete falso")

    st.write("### Probabilidades estimadas")
    st.write(f"Auténtico: {proba[0]:.2%}")
    st.write(f"Falso: {proba[1]:.2%}")\n