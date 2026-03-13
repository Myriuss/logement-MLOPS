import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Chargement des données
df = pd.read_csv("data/logements.csv")

# Variables
X = df[["surface", "pieces", "distance_centre", "etage", "annee_construction"]]
y = df["prix"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Évaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("MSE :", mse)
print("RMSE :", rmse)
print("R2 :", r2)

# Export du modèle
joblib.dump(model, "model/model.joblib")
print("Modèle exporté dans model/model.joblib")