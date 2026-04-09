import pickle
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load data
X_test = pickle.load(open("models/X_test.pkl", "rb"))
y_test = pickle.load(open("models/y_test.pkl", "rb"))

# Load model
model = pickle.load(open("models/final_model.pkl", "rb"))

print("Model Loaded Successfully")

# Prediction
y_pred = model.predict(X_test)

# -----------------------------
# 🔥 Accuracy
# -----------------------------
acc = accuracy_score(y_test, y_pred)
print(f"\n Accuracy: {acc:.3f}")

# -----------------------------
# 🔥 Classification Report
# -----------------------------
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 🔥 Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.show()
# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

plt.figure()
plt.bar(metrics, values)
plt.title("Model Evaluation Metrics")
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.show()