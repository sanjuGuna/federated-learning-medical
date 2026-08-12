from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os

from predict import predict_patient
from evaluate import get_evaluation_metrics

app = FastAPI(title="GAT+RDBN Multi-Dataset Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DiabetesPatientData(BaseModel):
    Age: float
    Gender: str
    Polyuria: str
    Polydipsia: str
    sudden_weight_loss: str
    weakness: str
    Polyphagia: str
    Genital_thrush: str
    visual_blurring: str
    Itching: str
    Irritability: str
    delayed_healing: str
    partial_paresis: str
    muscle_stiffness: str
    Alopecia: str
    Obesity: str

class HCVPatientData(BaseModel):
    Age: float
    Sex: str
    ALB: float
    ALP: float
    ALT: float
    AST: float
    BIL: float
    CHE: float
    CHOL: float
    CREA: float
    GGT: float
    PROT: float

class DermatologyPatientData(BaseModel):
    F1: float
    F2: float
    F3: float
    F4: float
    F5: float
    F6: float
    F7: float
    F8: float
    F9: float
    F10: float
    F11: float
    F12: float
    F13: float
    F14: float
    F15: float
    F16: float
    F17: float
    F18: float
    F19: float
    F20: float
    F21: float
    F22: float
    F23: float
    F24: float
    F25: float
    F26: float
    F27: float
    F28: float
    F29: float
    F30: float
    F31: float
    F32: float
    F33: float
    Age: float

@app.post("/predict/diabetes")
def predict_diabetes(patient: DiabetesPatientData):
    data_dict = patient.model_dump()
    mapped_dict = {
        "Age": data_dict["Age"],
        "Gender": data_dict["Gender"],
        "Polyuria": data_dict["Polyuria"],
        "Polydipsia": data_dict["Polydipsia"],
        "sudden weight loss": data_dict["sudden_weight_loss"],
        "weakness": data_dict["weakness"],
        "Polyphagia": data_dict["Polyphagia"],
        "Genital thrush": data_dict["Genital_thrush"],
        "visual blurring": data_dict["visual_blurring"],
        "Itching": data_dict["Itching"],
        "Irritability": data_dict["Irritability"],
        "delayed healing": data_dict["delayed_healing"],
        "partial paresis": data_dict["partial_paresis"],
        "muscle stiffness": data_dict["muscle_stiffness"],
        "Alopecia": data_dict["Alopecia"],
        "Obesity": data_dict["Obesity"]
    }
    return predict_patient(mapped_dict, dataset_name="diabetes")

@app.post("/predict/hcv")
def predict_hcv(patient: HCVPatientData):
    return predict_patient(patient.model_dump(), dataset_name="hcv")

@app.post("/predict/dermatology")
def predict_dermatology(patient: DermatologyPatientData):
    return predict_patient(patient.model_dump(), dataset_name="dermatology")

@app.get("/metrics")
def metrics(dataset: str = Query("diabetes")):
    return get_evaluation_metrics(dataset_name=dataset)

if not os.path.exists("frontend"):
    os.makedirs("frontend")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
