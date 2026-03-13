import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np

st.set_page_config(page_title="Dashboard ML", page_icon="📊", layout="wide")

st.title("📊 Dashboard — Analyse & Comparaison des modèles ML")
st.markdown("---")

# ─── Chargement des données ───────────────────────────────────────────────────
# parents[1] = racine du projet (pages/ -> racine)
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "logements.csv"
FALLBACK_PATH = Path("data/logements.csv")

@st.cache_data
def load_data():
    path = DATA_PATH if DATA_PATH.exists() else FALLBACK_PATH
    return pd.read_csv(path)

df = load_data()

# ─── 1. Aperçu des données ────────────────────────────────────────────────────
st.header("1. Aperçu des données")

col1, col2, col3 = st.columns(3)
col1.metric("Nombre de logements", len(df))
col2.metric("Variables", df.shape[1])
col3.metric("Prix moyen", f"{df['prix'].mean():,.0f} €")

with st.expander("📋 Voir le tableau complet", expanded=True):
    st.dataframe(df, use_container_width=True)

with st.expander("📈 Statistiques descriptives"):
    st.dataframe(df.describe().round(2), use_container_width=True)

st.markdown("---")

# ─── 2. Visualisations interactives ──────────────────────────────────────────
st.header("2. Visualisations interactives")

col_left, col_right = st.columns(2)

# Histogramme
with col_left:
    st.subheader("Histogramme")
    hist_col = st.selectbox("Variable", df.columns.tolist(), index=df.columns.tolist().index("prix"))
    fig_hist = px.histogram(
        df, x=hist_col, nbins=30,
        title=f"Distribution de « {hist_col} »",
        color_discrete_sequence=["#636EFA"],
        marginal="box"
    )
    fig_hist.update_layout(bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)

# Scatter plot
with col_right:
    st.subheader("Scatter plot")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    x_col = st.selectbox("Axe X", numeric_cols, index=numeric_cols.index("surface"))
    y_col = st.selectbox("Axe Y", numeric_cols, index=numeric_cols.index("prix"))
    color_col = st.selectbox("Couleur (optionnel)", ["Aucune"] + numeric_cols)
    fig_scatter = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=None if color_col == "Aucune" else color_col,
        trendline="ols",
        title=f"{x_col} vs {y_col}",
        color_continuous_scale="Viridis",
        opacity=0.7,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Matrice de corrélation
st.subheader("Matrice de corrélation")
corr = df.select_dtypes(include="number").corr().round(2)
fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Corrélation entre les variables",
    aspect="auto",
)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# ─── 3 & 4. Comparaison des modèles ML ────────────────────────────────────────
st.header("3. Comparaison des algorithmes de Machine Learning")

FEATURES = ["surface", "pieces", "distance_centre", "etage", "annee_construction"]
TARGET = "prix"

X = df[FEATURES]
y = df[TARGET]

col_params1, col_params2 = st.columns(2)
with col_params1:
    test_size = st.slider("Part des données de test", 0.1, 0.4, 0.2, 0.05)
with col_params2:
    random_state = st.number_input("Random state", value=42, min_value=0)

MODELS = {
    "Régression Linéaire": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "Arbre de Décision": DecisionTreeRegressor(random_state=random_state),
}

@st.cache_data
def train_and_evaluate(test_size, random_state):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    results = []
    predictions = {}
    for name, model in MODELS.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        mse = mean_squared_error(y_te, preds)
        results.append({
            "Modèle": name,
            "R²": round(r2_score(y_te, preds), 4),
            "MSE": round(mse, 2),
            "RMSE": round(np.sqrt(mse), 2),
        })
        predictions[name] = preds
    return pd.DataFrame(results), predictions, y_te

results_df, predictions, y_test_cached = train_and_evaluate(test_size, random_state)

# Tableau comparatif
st.subheader("4. Tableau comparatif des métriques")

best_r2_idx = results_df["R²"].idxmax()
styled = results_df.style.highlight_max(subset=["R²"], color="#d4edda") \
                         .highlight_min(subset=["MSE", "RMSE"], color="#d4edda") \
                         .format({"R²": "{:.4f}", "MSE": "{:,.2f}", "RMSE": "{:,.2f}"})
st.dataframe(styled, use_container_width=True, hide_index=True)

best_model = results_df.loc[best_r2_idx, "Modèle"]
st.success(f"🏆 Meilleur modèle (R²) : **{best_model}** — R² = {results_df.loc[best_r2_idx, 'R²']}")

# Graphiques comparatifs
col_m1, col_m2 = st.columns(2)

with col_m1:
    fig_r2 = px.bar(
        results_df, x="Modèle", y="R²",
        title="R² par modèle (plus c'est haut, mieux c'est)",
        color="R²", color_continuous_scale="Bluered",
        text_auto=".4f"
    )
    fig_r2.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig_r2, use_container_width=True)

with col_m2:
    fig_rmse = px.bar(
        results_df, x="Modèle", y="RMSE",
        title="RMSE par modèle (plus c'est bas, mieux c'est)",
        color="RMSE", color_continuous_scale="Reds",
        text_auto=".2f"
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

# Prédictions vs réalité
st.subheader("Prédictions vs Valeurs réelles")
selected_model = st.selectbox("Choisir un modèle à afficher", list(MODELS.keys()))

preds_selected = predictions[selected_model]
fig_pred = px.scatter(
    x=y_test_cached,
    y=preds_selected,
    labels={"x": "Prix réel (€)", "y": "Prix prédit (€)"},
    title=f"Prédictions vs Réalité — {selected_model}",
    opacity=0.7,
)
min_val = min(y_test_cached.min(), preds_selected.min())
max_val = max(y_test_cached.max(), preds_selected.max())
fig_pred.add_shape(
    type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
    line=dict(color="red", dash="dash"),
)
st.plotly_chart(fig_pred, use_container_width=True)
