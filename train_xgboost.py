import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
X_train = pickle.load(open("models/X_train.pkl", "rb"))
X_test = pickle.load(open("models/X_test.pkl", "rb"))
y_train = pickle.load(open("models/y_train.pkl", "rb"))
y_test = pickle.load(open("models/y_test.pkl", "rb"))

print("Data Loaded Successfully")

# Model
model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    eval_metric='mlogloss',
    use_label_encoder=False
)

model.fit(X_train, y_train)
print("XGBoost Model Trained Successfully")

# Prediction
pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, pred)
print(f"\nXGBoost Accuracy: {acc:.3f}")

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

# Save model
pickle.dump(model, open("models/xgb_model.pkl", "wb"))
print("\nModel Saved Successfully")