import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
# Load data
X_train = pickle.load(open("models/X_train.pkl", "rb"))
X_test = pickle.load(open("models/X_test.pkl", "rb"))
y_train = pickle.load(open("models/y_train.pkl", "rb"))
y_test = pickle.load(open("models/y_test.pkl", "rb"))
print("Data Loaded Successfully")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model Trained Successfully")
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("\nDecision Tree Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_test, pred))
# Save model
pickle.dump(model, open("models/dt_model.pkl", "wb"))
print("\nModel Saved Successfully")