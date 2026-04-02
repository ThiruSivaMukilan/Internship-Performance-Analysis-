import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load model
model = pickle.load(open("models/final_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Create API
app = FastAPI()

# Input structure
class InternData(BaseModel):
    attendance: float
    punctuality: float
    sprint_completion: float
    task_quality: float
    deadline_met: float
    communication: float
    collaboration: float
    initiative_score: float

# Test route
@app.get("/")
def home():
    return {"message": "Intern Performance API Live 🚀"}

@app.post("/predict")
def predict(data: InternData):

    try:
        input_data = np.array([[
            data.attendance,
            data.punctuality,
            data.sprint_completion,
            data.task_quality,
            data.deadline_met,
            data.communication,
            data.collaboration,
            data.initiative_score
        ]])

        print("INPUT:", input_data)

        input_scaled = scaler.transform(input_data)
        print("SCALED:", input_scaled)

        prediction = model.predict(input_scaled)[0]
        print("PRED:", prediction)

        if prediction == 0:
            result = "Poor"
        elif prediction == 1:
            result = "Average"
        else:
            result = "Good"

        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}