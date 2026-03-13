import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

st.set_page_config(page_title="MLOps Real Estate Dashboard", layout="wide")

st.title("🏠 Real Estate Price Prediction - ML Monitoring")

# -------------------
# LOAD DATA
# -------------------

df = pd.read_csv("../data/logements.csv")

# -------------------
# KPI
# -------------------

st.subheader("📊 Dataset Monitoring")

col1, col2, col3 = st.columns(3)

col1.metric("Nombre de logements", len(df))
col2.metric("Surface moyenne", round(df["surface"].mean(),1))
col3.metric("Prix moyen", round(df["prix"].mean(),0))

# -------------------
# GRAPHIQUES
# -------------------

st.subheader("📈 Analyse des données")

fig1 = px.scatter(
    df,
    x="surface",
    y="prix",
    title="Relation Surface / Prix",
)

st.plotly_chart(fig1, use_container_width=True)

fig2 = px.histogram(
    df,
    x="surface",
    title="Distribution des surfaces"
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------
# MACHINE LEARNING
# -------------------

st.subheader("🤖 Model Monitoring")

X = df[["surface","pieces","distance_centre","etage","annee_construction"]]
y = df["prix"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "KNN": KNeighborsRegressor(n_neighbors=3)
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    results.append({
        "Model": name,
        "MSE": mse,
        "R2": r2
    })

results_df = pd.DataFrame(results)

st.dataframe(results_df)

fig3 = px.bar(
    results_df,
    x="Model",
    y="R2",
    title="Comparaison performance modèles"
)

st.plotly_chart(fig3)

# -------------------
# PREDICTION
# -------------------

st.subheader("🏡 Prédire un prix")

surface = st.slider("Surface", 20, 200, 70)
pieces = st.slider("Nombre de pièces", 1, 8, 3)
distance = st.slider("Distance centre", 1, 20, 5)
etage = st.slider("Etage", 0, 20, 3)
annee = st.slider("Année construction", 1950, 2025, 2015)

input_data = pd.DataFrame({
    "surface":[surface],
    "pieces":[pieces],
    "distance_centre":[distance],
    "etage":[etage],
    "annee_construction":[annee]
})

best_model = RandomForestRegressor()

best_model.fit(X_train, y_train)

prediction = best_model.predict(input_data)

st.success(f"💰 Prix estimé : {int(prediction[0])} €")