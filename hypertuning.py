import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load data
X_train = pickle.load(open("models/X_train.pkl", "rb"))
y_train = pickle.load(open("models/y_train.pkl", "rb"))
X_test = pickle.load(open("models/X_test.pkl", "rb"))
y_test = pickle.load(open("models/y_test.pkl", "rb"))

print("Data Loaded Successfully")

# Base model
rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

# 🔥 BIGGER SEARCH SPACE
param_dist = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ['sqrt', 'log2', None],
    "bootstrap": [True, False]
}

# 🔥 RANDOM SEARCH (BETTER THAN GRID)
search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=25,              # more combinations
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42
)

print(" Hyperparameter tuning started...")

search.fit(X_train, y_train)

print("Best Parameters:", search.best_params_)

best_model = search.best_estimator_

# Prediction
y_pred = best_model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

print("\n Final Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
pickle.dump(best_model, open("models/final_model.pkl", "wb"))

print("\n Tuned Model Saved Successfully")