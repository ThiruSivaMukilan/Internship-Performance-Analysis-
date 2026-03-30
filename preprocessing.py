import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("data/processed_intern_data.csv")

print("Loaded Data:")
print(df.head())

# -------------------------------
# Handle missing values
# -------------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)

# -------------------------------
# Features & Target
# -------------------------------
X = df.drop("performance", axis=1)
y = df["performance"]

# 🔥 FIXED LABEL MAPPING
y = y.map({
    "low": 0,
    "medium": 1,
    "high": 2
})

# Check mapping
if y.isnull().sum() > 0:
    raise ValueError("Target contains unknown labels!")

print("\nDataset Info:")
print(df.info())

print("\nTarget Distribution:")
print(y.value_counts())

print("\nFeature Columns:", X.columns.tolist())

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nSplit Done")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nScaling Completed")

# -------------------------------
# Save files
# -------------------------------
os.makedirs("models", exist_ok=True)

pickle.dump(X_train, open("models/X_train.pkl", "wb"))
pickle.dump(X_test, open("models/X_test.pkl", "wb"))
pickle.dump(y_train, open("models/y_train.pkl", "wb"))
pickle.dump(y_test, open("models/y_test.pkl", "wb"))

pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("models/feature_names.pkl", "wb"))

print("\n✅ All files saved successfully")