from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="API Prix Logement")

model = joblib.load("model/model.joblib") #mettre le bon chemin

class LogementInput(BaseModel):
    surface: float
    pieces: int
    distance_centre: float
    etage: int
    annee_construction: int

@app.get("/")
def home():
    return {"message": "API de prédiction des prix de logements"}

@app.post("/predict")
def predict(data: LogementInput):
    df = pd.DataFrame([{
        "surface": data.surface,
        "pieces": data.pieces,
        "distance_centre": data.distance_centre,
        "etage": data.etage,
        "annee_construction": data.annee_construction
    }])

    prediction = model.predict(df)[0]

    return {
        "prix_estime": float(round(prediction, 2))
    }