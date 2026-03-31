import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load Data
X_train = pickle.load(open("models/X_train.pkl", "rb"))
X_test = pickle.load(open("models/X_test.pkl", "rb"))
y_train = pickle.load(open("models/y_train.pkl", "rb"))
y_test = pickle.load(open("models/y_test.pkl", "rb"))
print("Data Loaded Successfully")
# Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)
model.fit(X_train, y_train)
print("Random Forest Model Trained Successfully")
# Prediction
pred = model.predict(X_test)
# Evaluation
acc = accuracy_score(y_test, pred)
print(f"\nRandom Forest Accuracy: {acc:.3f}")
print("\nClassification Report:\n")
print(classification_report(y_test, pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
# Save Model
pickle.dump(model, open("models/rf_model.pkl", "wb"))
print("\nModel Saved Successfully")