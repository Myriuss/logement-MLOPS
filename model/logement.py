import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


df = pd.read_csv("logements.csv")
print("Affiché les données du dataset")
print(df)

print("\nAffiché les 5 premieres lignes du dataset : ")
print(df.head())
print("\n-----------------Analyse de données---------------------\n")
print("\nAffiché le nombre de non null et type de données(texte, nombre)qu'on a dans chaque colonne : \n")
print(df.info())
print("\nDeterminé la pertinance d'une donnée\n")
print(df.describe())
print("\nCalculer le nombre de null qu'on a dans une colonne et le nombre de données dupliquer\n")
print(df.isnull().sum())
print(df.duplicated().sum())

print("\nAffichage d'un diagrame sur axe x(surface) et y(prix)\n")
plt.scatter(df["surface"], df["prix"])
plt.xlabel("Surface")
plt.ylabel("Prix")
plt.title("Relation entre la surface et le prix")
plt.show()
print("\n-----------------Entrainement de la machine learning---------------------\n")

X = df[["surface","pieces","distance_centre","etage","annee_construction"]]
y = df["prix"]

# Création de variable pour entraianement et pour test (20% des données sont pour le test et 80% pour l'entrainement)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Création de la variable executant l'algorythme
model = LinearRegression()

# Entrainement de la machine learning avec la variable X_train(variable entrés) y_train(variable cible)
model.fit(X_train,y_train)

# Tenter une prédiction en utilisant les 20% de données test
predictions = model.predict(X_test)

print("\nAffichage des resultat de prédiction\n")
resultats = pd.DataFrame({
    "prix_reel": y_test.values,
    "prix_predit": predictions
})

print(resultats)

print("\nEvaluation des performances\n")
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MSE :", mse)
print("R2 :", r2)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("Score train :", train_score)
print("Score test :", test_score)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("RMSE :", rmse)

print("\nPrédire deux nouveaux logements\n")
nouveau_logement = pd.DataFrame({
    "surface":[78,9],
    "pieces":[3,1],
    "distance_centre":[5,3],
    "etage":[4,3],
    "annee_construction":[2016,2022]
})

prediction = model.predict(nouveau_logement)

print("Prix estimé pour le 78 :", prediction[0])
print("Prix estimé pour le 9^2:", prediction[1])

print("\nPrédiction avec différents algorythme\n")
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=3)
}

for name, model in models.items():
    print("\n----------------------------------")
    print("Algorithme :", name)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("MSE :", mse)
    print("R2 :", r2)
