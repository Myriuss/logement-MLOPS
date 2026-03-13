from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prédiction logement", page_icon="🏠")

st.title("🏠 Estimation du prix d'un logement")
st.write("Saisissez les caractéristiques du logement pour obtenir une estimation.")

# Chargement du modèle directement (sans API)
MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Formulaire
surface = st.number_input("Surface (m²)", min_value=1.0, value=75.0)
pieces = st.number_input("Nombre de pièces", min_value=1, value=3)
distance_centre = st.number_input("Distance au centre (km)", min_value=0.0, value=5.0)
etage = st.number_input("Étage", min_value=0, value=2)
annee_construction = st.number_input(
    "Année de construction", min_value=1900, max_value=2100, value=2015
)

if st.button("Prédire le prix"):
    data = pd.DataFrame([{
        "surface": surface,
        "pieces": pieces,
        "distance_centre": distance_centre,
        "etage": etage,
        "annee_construction": annee_construction,
    }])
    prix = model.predict(data)[0]
    st.success(f"Prix estimé : {prix:,.2f} €")
