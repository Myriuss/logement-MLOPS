from pydantic import BaseModel, Field


class LogementInput(BaseModel):
    surface: float = Field(..., gt=0, description="Surface en m²")
    pieces: int = Field(..., ge=1, description="Nombre de pièces")
    distance_centre: float = Field(..., ge=0, description="Distance au centre-ville en km")
    etage: int = Field(..., ge=0, description="Étage")
    annee_construction: int = Field(..., ge=1800, le=2100, description="Année de construction")


class PredictionResponse(BaseModel):
    prix_estime: float
