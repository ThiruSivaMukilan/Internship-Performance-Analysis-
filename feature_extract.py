import pandas as pd
import numpy as np
import os  

df = pd.read_csv("data/real_world_intern_data.csv")

print("Dataset Loaded Successfully")
print(df.head())

df.fillna(df.mean(numeric_only=True), inplace=True)

def extract_features(row):
    sprint_completion = (
        (row['sprints_done'] / row['sprints_total']) * 100
        if row['sprints_total'] != 0 else 0
    )

    task_quality = row['code_review_score']
    deadline_met = row['deadline_met_percentage']
    attendance = row['attendance_percentage']
    punctuality = row['punctuality'] * 100

    communication = (
        (row['meetings_attended'] / row['meetings_total']) * 100
        if row['meetings_total'] != 0 else 0
    )

    collaboration = communication * 0.9

    initiative_score = (
        (row['tasks_completed'] / row['tasks_assigned']) * 100
        if row['tasks_assigned'] != 0 else 0
    )

    return [
        attendance,
        punctuality,
        sprint_completion,
        task_quality,
        deadline_met,
        communication,
        collaboration,
        initiative_score
    ]

processed_data = []

for _, row in df.iterrows():
    processed_data.append(extract_features(row))

columns = [
    "attendance",
    "punctuality",
    "sprint_completion",
    "task_quality",
    "deadline_met",
    "communication",
    "collaboration",
    "initiative_score"
]

df_features = pd.DataFrame(processed_data, columns=columns)

# Use existing label from dataset
df_features["performance"] = df["performance"]

print("\nProcessed Dataset Preview:")
print(df_features.head())

print("\nClass Distribution:")
print(df_features["performance"].value_counts())

os.makedirs("data", exist_ok=True)
df_features.to_csv("data/processed_intern_data.csv", index=False)

print("\n✅ Processed dataset saved successfully!")