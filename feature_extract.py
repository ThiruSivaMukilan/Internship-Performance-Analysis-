# feature_engineering.py

import pandas as pd
import numpy as np
import os  

df = pd.read_csv("data/real_world_intern_data.csv")

df.fillna(df.mean(numeric_only=True), inplace=True)

def extract_features(row):
    sprint_completion = (row['sprints_done'] / row['sprints_total']) * 100 if row['sprints_total'] != 0 else 0
    task_quality = row['code_review_score']
    deadline_met = row['deadline_met_percentage']
    attendance = row['attendance_percentage']
    punctuality = row['punctuality'] * 100

    communication = (row['meetings_attended'] / row['meetings_total']) * 100 if row['meetings_total'] != 0 else 0
    collaboration = communication * 0.9

    initiative_score = (row['tasks_completed'] / row['tasks_assigned']) * 100 if row['tasks_assigned'] != 0 else 0

    return [
        attendance, punctuality, sprint_completion, task_quality,
        deadline_met, communication, collaboration, initiative_score
    ]

processed_data = [extract_features(row) for _, row in df.iterrows()]

columns = [
    "attendance","punctuality","sprint_completion","task_quality",
    "deadline_met","communication","collaboration","initiative_score"
]

df_features = pd.DataFrame(processed_data, columns=columns)

# IMPORTANT FEATURES ONLY (NO OVERLOAD)
df_features["work_quality"] = (df_features["task_quality"] + df_features["deadline_met"]) / 2
df_features["productivity"] = (df_features["sprint_completion"] + df_features["initiative_score"]) / 2
df_features["performance_score"] = (
    0.3 * df_features["sprint_completion"] +
    0.3 * df_features["task_quality"] +
    0.2 * df_features["deadline_met"] +
    0.2 * df_features["attendance"]
)

# TARGET
df_features.insert(0, "intern_id", df["intern_id"])
df_features["performance"] = df["performance"]

os.makedirs("data", exist_ok=True)
df_features.to_csv("data/processed_intern_data.csv", index=False)

print("Feature Engineering Completed")