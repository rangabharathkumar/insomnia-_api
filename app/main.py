from fastapi import FastAPI
from pydantic import BaseModel
from app.model_utils import predict

# Create the FastAPI instance
app = FastAPI(title="Insomnia Prediction API")

# Input data schema using Pydantic for validation
class InputData(BaseModel):
    Gender: str
    Age: int
    Occupation: str
    Sleep_Duration: float
    Quality_of_Sleep: int
    Physical_Activity_Level: int
    Stress_Level: int
    BMI_Category: str
    Blood_Pressure: str
    Heart_Rate: int
    Daily_Steps: int

# Endpoint for prediction
@app.post("/predict")
def get_prediction(data: InputData):
    """Endpoint to get the prediction for insomnia condition."""
    input_data = data.dict()
    result = predict(input_data)
    return {"prediction": result}
