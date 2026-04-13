from fastapi import FastAPI, Request, Form, Body
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import pickle
import numpy as np
import json
import os

app = FastAPI()

# Templates
templates = Jinja2Templates(directory="templates")

# Load model & scaler
model = pickle.load(open("models/final_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Load dataset
df = pd.read_csv("data/processed_intern_data.csv")

# ---------------------------
# USER MANAGEMENT (JSON)
# ---------------------------
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {"admin": "admin"}  # demo login

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# ---------------------------
# LOGIN PAGE
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")

# ---------------------------
# LOGIN POST
# ---------------------------
@app.post("/login", response_class=HTMLResponse)
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    
    users = load_users()

    if username in users and users[username] == password:
        return templates.TemplateResponse(request=request, name="dashboard.html", context={
            "interns": df.to_dict(orient="records")
        })

    return templates.TemplateResponse(request=request, name="login.html", context={
        "error": "Invalid Username or Password"
    })

# ---------------------------
# REGISTER PAGE
# ---------------------------
@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(request=request, name="register.html")

# ---------------------------
# REGISTER POST
# ---------------------------
@app.post("/register", response_class=HTMLResponse)
def register(request: Request, company_name: str = Form(...), username: str = Form(...), password: str = Form(...)):
    
    users = load_users()

    if username in users:
        return templates.TemplateResponse(request=request, name="register.html", context={
            "error": "Username already exists!"
        })

    users[username] = password
    save_users(users)

    return templates.TemplateResponse(request=request, name="login.html", context={
        "success": f"Registered successfully for {company_name}"
    })

# ---------------------------
# DASHBOARD
# ---------------------------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html", context={
        "interns": df.to_dict(orient="records")
    })

# ---------------------------
# INTERN DETAIL
# ---------------------------
@app.get("/intern/{intern_id}", response_class=HTMLResponse)
def intern_detail(request: Request, intern_id: str):

    intern_df = df[df["intern_id"] == intern_id]

    if intern_df.empty:
        return templates.TemplateResponse(request=request, name="dashboard.html", context={
            "interns": df.to_dict(orient="records"),
            "error": "Intern not found"
        })

    intern = intern_df.iloc[0]

    # Feature Engineering
    work_quality = (intern["task_quality"] + intern["deadline_met"]) / 2
    productivity = (intern["sprint_completion"] + intern["initiative_score"]) / 2

    performance_score = (
        0.25 * intern["sprint_completion"] +
        0.25 * intern["task_quality"] +
        0.2 * intern["deadline_met"] +
        0.1 * intern["attendance"] +
        0.1 * intern["initiative_score"] +
        0.1 * intern["communication"]
    )

    input_data = np.array([[  
        intern["attendance"],
        intern["punctuality"],
        intern["sprint_completion"],
        intern["task_quality"],
        intern["deadline_met"],
        intern["communication"],
        intern["collaboration"],
        intern["initiative_score"],
        work_quality,
        productivity,
        performance_score
    ]])

    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)[0]
    result = ["Low", "Medium", "High"][pred]

    return templates.TemplateResponse(request=request, name="detail.html", context={
        "intern": intern.to_dict(),
        "prediction": result
    })

# ---------------------------
# 🔥 PREDICTION API (SPRINT 4)
# ---------------------------
@app.post("/predict")
def predict_api(data: dict = Body(...)):

    input_data = np.array([[
        data["attendance"],
        data["punctuality"],
        data["sprint_completion"],
        data["task_quality"],
        data["deadline_met"],
        data["communication"],
        data["collaboration"],
        data["initiative_score"],
        data["work_quality"],
        data["productivity"],
        data["performance_score"]
    ]])

    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)[0]
    result = ["Low", "Medium", "High"][pred]

    return {
        "prediction": result
    }