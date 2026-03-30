import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Load Data
# -------------------------------
X_train = pickle.load(open("models/X_train.pkl", "rb"))
y_train = pickle.load(open("models/y_train.pkl", "rb"))

X_test = pickle.load(open("models/X_test.pkl", "rb"))
y_test = pickle.load(open("models/y_test.pkl", "rb"))

print("Data Loaded Successfully")

# -------------------------------
# Base Model
# -------------------------------
rf = RandomForestClassifier(random_state=42)

# -------------------------------
# Parameter Grid (Improved)
# -------------------------------
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# -------------------------------
# Grid Search
# -------------------------------
grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,              # better validation
    scoring='accuracy',
    n_jobs=-1
)

# -------------------------------
# Train
# -------------------------------
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# -------------------------------
# Best Model
# -------------------------------
best_model = grid.best_estimator_

# -------------------------------
# Evaluate
# -------------------------------
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nImproved Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Save Final Model
# -------------------------------
pickle.dump(best_model, open("models/final_model.pkl", "wb"))

print("\n✅ Final Model Saved Successfully")