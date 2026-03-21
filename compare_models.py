import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# -----------------------------------
# Load Test Data
# -----------------------------------
X_test = pickle.load(open("models/X_test.pkl", "rb"))
y_test = pickle.load(open("models/y_test.pkl", "rb"))

# -----------------------------------
# Load Models
# -----------------------------------
dt = pickle.load(open("models/dt_model.pkl", "rb"))
rf = pickle.load(open("models/rf_model.pkl", "rb"))
xgb = pickle.load(open("models/xgb_model.pkl", "rb"))

print("Models Loaded Successfully")

# -----------------------------------
# Predictions
# -----------------------------------
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)

# -----------------------------------
# Accuracy
# -----------------------------------
dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

print("\nAccuracy Comparison:")
print(f"Decision Tree: {dt_acc:.3f}")
print(f"Random Forest: {rf_acc:.3f}")
print(f"XGBoost: {xgb_acc:.3f}")

# -----------------------------------
# Visualization (FIXED 🔥)
# -----------------------------------
plt.figure()

models = ["Decision Tree", "Random Forest", "XGBoost"]
accuracy = [dt_acc, rf_acc, xgb_acc]

plt.bar(models, accuracy)

# 🔥 IMPORTANT FIX (dynamic range)
plt.ylim(min(accuracy) - 0.02, 1.0)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison (Improved View)")

# Show values clearly
for i, v in enumerate(accuracy):
    plt.text(i, v + 0.003, f"{v:.3f}", ha='center', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()