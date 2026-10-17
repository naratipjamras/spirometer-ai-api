from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).resolve().parent / "spirometry_model.joblib"
EXPECTED_FEATURES = [
    "age",
    "height",
    "weight",
    "FVC",
    "FEV1",
    "PEF",
    "FEV1_PER_FVC",
]

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
label_encoder = bundle["label_encoder"]
features = bundle["features"]
if features != EXPECTED_FEATURES:
    raise ValueError(
        f"Unexpected model features in {MODEL_PATH}: {features!r}"
    )

app = FastAPI(
    title="Spirometry Diagnosis API",
    description="Predicts a spirometry diagnosis using the saved scikit-learn model.",
    version="1.0.0",
)


class PatientInput(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    age: float = Field(ge=0)
    height: float = Field(gt=0)
    weight: float = Field(gt=0)
    fvc: float = Field(gt=0)
    fev1: float = Field(ge=0)
    pef: float = Field(ge=0)


class PredictionResponse(BaseModel):
    diagnosis: str
    probabilities: dict[str, float]


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_diagnosis(patient: PatientInput) -> PredictionResponse:
    row = pd.DataFrame(
        [
            [
                patient.age,
                patient.height,
                patient.weight,
                patient.fvc,
                patient.fev1,
                patient.pef,
                patient.fev1 / patient.fvc,
            ]
        ],
        columns=features,
    )

    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    diagnosis = label_encoder.inverse_transform([prediction])[0]

    return PredictionResponse(
        diagnosis=str(diagnosis),
        probabilities={
            str(label): float(probability)
            for label, probability in zip(label_encoder.classes_, probabilities)
        },
    )
