from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from api.schemas import LogementInput, PredictionResponse

app = FastAPI(title="API Prix Logement", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "API de prédiction des prix de logements",
        "model_path": str(MODEL_PATH),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: LogementInput):
    df = pd.DataFrame(
        [
            {
                "surface": data.surface,
                "pieces": data.pieces,
                "distance_centre": data.distance_centre,
                "etage": data.etage,
                "annee_construction": data.annee_construction,
            }
        ]
    )

    prediction = model.predict(df)[0]
    return PredictionResponse(prix_estime=float(round(prediction, 2)))
