from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Change this to "decision_tree_model.joblib" for your second app
model = joblib.load("decision_tree_model.joblib")

# List of columns in the EXACT order from your notebook
COLUMNS = [
    'age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 
    'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
    'restecg_normal', 'restecg_st-t abnormality', 'slope_flat',
    'slope_upsloping', 'thal_normal', 'thal_reversable defect', 
    'sex_Male', 'fbs_True', 'exang_True'
]

class PatientData(BaseModel):
    age: float; trestbps: float; chol: float; thalch: float; oldpeak: float; ca: float; sex: int
    cp: str; restecg: str; slope: str; thal: str; fbs: int; exang: int

@app.get("/", response_class=HTMLResponse)
async def get_form():
    with open("index.html") as f:
        return f.read()

@app.post("/predict")
async def predict(data: PatientData):
    # Initialize all columns as 0
    input_row = {col: 0 for col in COLUMNS}
    
    # Fill numeric values
    input_row['age'] = data.age
    input_row['trestbps'] = data.trestbps
    input_row['chol'] = data.chol
    input_row['thalch'] = data.thalch
    input_row['oldpeak'] = data.oldpeak
    input_row['ca'] = data.ca
    input_row['sex_Male'] = data.sex
    input_row['fbs_True'] = data.fbs
    input_row['exang_True'] = data.exang

    # Map categorical selections to columns
    if f"cp_{data.cp}" in input_row: input_row[f"cp_{data.cp}"] = 1
    if f"restecg_{data.restecg}" in input_row: input_row[f"restecg_{data.restecg}"] = 1
    if f"slope_{data.slope}" in input_row: input_row[f"slope_{data.slope}"] = 1
    if f"thal_{data.thal}" in input_row: input_row[f"thal_{data.thal}"] = 1

    # Create DataFrame with forced column order
    df = pd.DataFrame([input_row])[COLUMNS]
    
    prediction = model.predict(df)[0]
    result_text = "Heart Disease Likely" if prediction == 1 else "No Heart Disease Detected"
    return {"prediction": result_text, "color": "#d9534f" if prediction == 1 else "#28a745"}