"""
Spirometry Diagnosis API (FastAPI)

Requirements:
    pip install fastapi uvicorn scikit-learn pandas joblib

Run:
    uvicorn main:app --reload

Docs (Swagger UI):
    http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

MODEL_PATH = "spirometry_model.joblib"


# ----------------------------------------------------------------------
# Load model once at startup (not on every request)
# ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(MODEL_PATH)
    app.state.model = bundle["model"]
    app.state.le = bundle["label_encoder"]
    app.state.features = bundle["features"]
    yield


app = FastAPI(
    title="Spirometry Diagnosis API",
    description="Predict Normal / Obstructive / Restrictive from spirometry values.",
    version="1.0.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------
# Request / response schemas
# ----------------------------------------------------------------------
class PatientInput(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    height: float = Field(..., gt=0, le=250, description="Height in cm")
    weight: float = Field(..., gt=0, le=400, description="Weight in kg")
    FVC: float = Field(..., gt=0, le=10, description="Forced vital capacity (L)")
    FEV1: float = Field(..., gt=0, le=10, description="Forced expiratory volume in 1 s (L)")
    PEF: float = Field(..., gt=0, le=20, description="Peak expiratory flow (L/s)")

    @model_validator(mode="after")
    def fev1_not_greater_than_fvc(self):
        if self.FEV1 > self.FVC:
            raise ValueError("FEV1 cannot be greater than FVC")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 55,
                "height": 168,
                "weight": 72,
                "FVC": 3.8,
                "FEV1": 2.0,
                "PEF": 4.2,
            }
        }
    }


class PredictionResponse(BaseModel):
    diagnosis: str
    fev1_per_fvc: float
    probabilities: dict[str, float] | None = None


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    model = app.state.model
    le = app.state.le
    features = app.state.features

    fev1_per_fvc = patient.FEV1 / patient.FVC

    row = {
        "age": patient.age,
        "height": patient.height,
        "weight": patient.weight,
        "FVC": patient.FVC,
        "FEV1": patient.FEV1,
        "PEF": patient.PEF,
        "FEV1_PER_FVC": fev1_per_fvc,
    }
    # Build DataFrame using the saved feature order
    X = pd.DataFrame([[row[f] for f in features]], columns=features)

    try:
        pred = model.predict(X)[0]
        diagnosis = le.inverse_transform([pred])[0]

        probabilities = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            class_names = le.inverse_transform(model.classes_)
            probabilities = {
                str(name): round(float(p), 4) for name, p in zip(class_names, proba)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        diagnosis=str(diagnosis),
        fev1_per_fvc=round(fev1_per_fvc, 3),
        probabilities=probabilities,
    )