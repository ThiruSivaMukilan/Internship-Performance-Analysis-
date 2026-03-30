import pandas as pd
import numpy as np
import os

np.random.seed(42)

rows = 5000
data = []

os.makedirs("data", exist_ok=True)

for i in range(rows):
    intern_id = f"RX{str(i+1).zfill(4)}"

    # Balanced performance selection
    perf_type = np.random.choice(['high', 'medium', 'low'], p=[0.3, 0.4, 0.3])

    # -------------------------------
    # REALISTIC OVERLAPPING DATA
    # -------------------------------

    # Common base values (shared randomness)
    base_sprint = np.random.randint(4, 8)
    base_tasks = np.random.randint(8, 15)
    base_meetings = np.random.randint(15, 25)

    if perf_type == 'high':
        sprints_total = base_sprint
        sprints_done = np.random.randint(int(base_sprint*0.7), base_sprint+1)

        tasks_assigned = base_tasks
        tasks_completed = np.random.randint(int(base_tasks*0.7), base_tasks+1)

        meetings_total = base_meetings
        meetings_attended = np.random.randint(int(base_meetings*0.7), base_meetings+1)

        code_review_score = np.random.normal(70, 20)
        deadline_met = np.random.normal(75, 20)
        attendance = np.random.normal(80, 20)

        punctuality = np.random.choice([0, 1], p=[0.2, 0.8])

    elif perf_type == 'medium':
        sprints_total = base_sprint
        sprints_done = np.random.randint(int(base_sprint*0.5), base_sprint)

        tasks_assigned = base_tasks
        tasks_completed = np.random.randint(int(base_tasks*0.5), base_tasks)

        meetings_total = base_meetings
        meetings_attended = np.random.randint(int(base_meetings*0.5), base_meetings)

        code_review_score = np.random.normal(60, 25)
        deadline_met = np.random.normal(65, 25)
        attendance = np.random.normal(65, 25)

        punctuality = np.random.choice([0, 1])

    else:  # low
        sprints_total = base_sprint
        sprints_done = np.random.randint(1, int(base_sprint*0.7))

        tasks_assigned = base_tasks
        tasks_completed = np.random.randint(1, int(base_tasks*0.7))

        meetings_total = base_meetings
        meetings_attended = np.random.randint(1, int(base_meetings*0.7))

        code_review_score = np.random.normal(50, 30)
        deadline_met = np.random.normal(55, 30)
        attendance = np.random.normal(55, 30)

        punctuality = np.random.choice([0, 1], p=[0.7, 0.3])

    # -------------------------------
    # EXTRA RANDOMNESS (KEY FIX 🔥)
    # -------------------------------
    if np.random.rand() < 0.1:
        # flip behavior randomly (real-world inconsistency)
        code_review_score += np.random.normal(0, 30)
        attendance += np.random.normal(0, 30)

    # Clip values
    code_review_score = np.clip(code_review_score, 0, 100)
    deadline_met = np.clip(deadline_met, 0, 100)
    attendance = np.clip(attendance, 0, 100)

    # Missing values
    if np.random.rand() < 0.05:
        code_review_score = None
    if np.random.rand() < 0.05:
        attendance = None

    data.append([
        intern_id,
        sprints_done,
        sprints_total,
        tasks_completed,
        tasks_assigned,
        meetings_attended,
        meetings_total,
        None if code_review_score is None else round(code_review_score, 2),
        round(deadline_met, 2),
        None if attendance is None else round(attendance, 2),
        punctuality,
        perf_type
    ])

# Columns
columns = [
    "intern_id",
    "sprints_done",
    "sprints_total",
    "tasks_completed",
    "tasks_assigned",
    "meetings_attended",
    "meetings_total",
    "code_review_score",
    "deadline_met_percentage",
    "attendance_percentage",
    "punctuality",
    "performance"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("data/real_world_intern_data.csv", index=False)

print("✅ Realistic dataset created successfully!")
print(df.head())
print("\nClass Distribution:")
print(df["performance"].value_counts())