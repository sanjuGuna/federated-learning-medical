from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_new_patient

app = FastAPI(title="GAT+RDBN Diabetes Prediction API")

class PatientData(BaseModel):
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

@app.post("/predict")
def predict(patient: PatientData):
    data_dict = patient.dict()
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
    
    result = predict_new_patient(mapped_dict)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
