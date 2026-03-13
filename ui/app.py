import os

import requests
import streamlit as st

st.set_page_config(page_title="Prédiction logement", page_icon="🏠")

st.title("Estimation du prix d’un logement")
st.write("Saisissez les caractéristiques du logement pour obtenir une estimation.")

surface = st.number_input("Surface (m²)", min_value=1.0, value=75.0)
pieces = st.number_input("Nombre de pièces", min_value=1, value=3)
distance_centre = st.number_input("Distance au centre (km)", min_value=0.0, value=5.0)
etage = st.number_input("Étage", min_value=0, value=2)
annee_construction = st.number_input(
    "Année de construction", min_value=1900, max_value=2100, value=2015
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
st.caption(f"API utilisée : {API_URL}")

if st.button("Prédire le prix"):
    payload = {
        "surface": surface,
        "pieces": pieces,
        "distance_centre": distance_centre,
        "etage": etage,
        "annee_construction": annee_construction,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prix estimé : {result['prix_estime']} €")
        else:
            st.error(f"Erreur API : {response.status_code}")
            st.write(response.text)

    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de contacter l’API : {e}")
