import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data
X_train = pickle.load(open("models/X_train.pkl", "rb"))
y_train = pickle.load(open("models/y_train.pkl", "rb"))
X_test = pickle.load(open("models/X_test.pkl", "rb"))
y_test = pickle.load(open("models/y_test.pkl", "rb"))

print("✅ Data Loaded Successfully")

# Base model
rf = RandomForestClassifier(random_state=42)

# Improved parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Grid Search
grid = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

print(" Starting Hyperparameter Tuning...")

# Train
grid.fit(X_train, y_train)

print(" Best Parameters:", grid.best_params_)

# Best model
best_model = grid.best_estimator_

# Predict
y_pred = best_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\n Tuned Model Accuracy:", round(accuracy, 3))

# Classification report
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Tuned Model")
plt.show()

# Save final model
pickle.dump(best_model, open("models/final_model.pkl", "wb"))

print("\n Final Model Saved Successfully!")