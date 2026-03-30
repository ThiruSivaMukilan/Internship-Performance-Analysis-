import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# Load Data
# -------------------------------
X_test = pickle.load(open("models/X_test.pkl", "rb"))
y_test = pickle.load(open("models/y_test.pkl", "rb"))

# Load Models
dt = pickle.load(open("models/dt_model.pkl", "rb"))
rf = pickle.load(open("models/rf_model.pkl", "rb"))
xgb = pickle.load(open("models/xgb_model.pkl", "rb"))

print("Models Loaded Successfully")

# -------------------------------
# Predictions
# -------------------------------
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)

# -------------------------------
# Accuracy
# -------------------------------
dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

print("\n🔹 Accuracy Comparison:")
print(f"Decision Tree: {dt_acc:.3f}")
print(f"Random Forest: {rf_acc:.3f}")
print(f"XGBoost: {xgb_acc:.3f}")

# -------------------------------
# Classification Reports
# -------------------------------
print("\n🔹 Decision Tree Report:\n", classification_report(y_test, dt_pred))
print("\n🔹 Random Forest Report:\n", classification_report(y_test, rf_pred))
print("\n🔹 XGBoost Report:\n", classification_report(y_test, xgb_pred))

# -------------------------------
# Confusion Matrix (Best Model)
# -------------------------------
best_model_name = max(
    [("DT", dt_acc), ("RF", rf_acc), ("XGB", xgb_acc)],
    key=lambda x: x[1]
)[0]

if best_model_name == "DT":
    best_pred = dt_pred
elif best_model_name == "RF":
    best_pred = rf_pred
else:
    best_pred = xgb_pred

print(f"\n🏆 Best Model: {best_model_name}")

cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()

# -------------------------------
# Accuracy Graph
# -------------------------------
plt.figure()

models = ["Decision Tree", "Random Forest", "XGBoost"]
accuracy = [dt_acc, rf_acc, xgb_acc]

plt.bar(models, accuracy)
plt.ylim(min(accuracy) - 0.05, 1.0)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison")

# Add values on top
for i, v in enumerate(accuracy):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()