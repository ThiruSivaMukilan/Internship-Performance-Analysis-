import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
# Load data
X_train = pickle.load(open("models/X_train.pkl", "rb"))
X_test = pickle.load(open("models/X_test.pkl", "rb"))
y_train = pickle.load(open("models/y_train.pkl", "rb"))
y_test = pickle.load(open("models/y_test.pkl", "rb"))
print("Data Loaded Successfully")
# Train model
model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)
print("Model Trained Successfully")
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("\nXGBoost Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_test, pred))
# Save model
pickle.dump(model, open("models/xgb_model.pkl", "wb"))
print("\nModel Saved Successfully")