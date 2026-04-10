from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
import pandas as pd
import pickle
import numpy as np
app = FastAPI()
# Templates
templates = Jinja2Templates(directory="templates")
# Load model & scaler
model = pickle.load(open("models/final_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Load dataset
df = pd.read_csv("data/processed_intern_data.csv")

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

    if username == "admin" and password == "admin":
        return templates.TemplateResponse(request=request, name="dashboard.html", context={
            "interns": df.to_dict(orient="records")
        })

    return templates.TemplateResponse(request=request, name="login.html", context={
        "error": "Invalid Username or Password"
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

    # ❌ safety check
    if intern_df.empty:
        return templates.TemplateResponse(request=request, name="dashboard.html", context={
            "interns": df.to_dict(orient="records"),
            "error": "Intern not found"
        })

    intern = intern_df.iloc[0]

    # 🔥 Feature Engineering (MUST MATCH TRAINING)
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

    # 🔥 Model input (order MUST MATCH training)
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

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    pred = model.predict(input_scaled)[0]

    result = ["Low", "Medium", "High"][pred]

    return templates.TemplateResponse(request=request, name="detail.html", context={
        "intern": intern.to_dict(),   # 🔥 IMPORTANT FIX
        "prediction": result
    })