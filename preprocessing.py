import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
df = pd.read_csv("data/processed_intern_data.csv")
print("Loaded Data:")
print(df.head())
df.fillna(df.mean(numeric_only=True), inplace=True)
X = df.drop("performance", axis=1)
y = df["performance"]
y = y.map({
    "Poor": 0,
    "Average": 1,
    "Good": 2
})
if y.isnull().sum() > 0:
    raise ValueError("Target contains unknown labels!")
print("\nDataset Info:")
print(df.info())
print("\nTarget Distribution:")
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nSplit Done")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\nScaling Completed")
os.makedirs("models", exist_ok=True)
pickle.dump(X_train, open("models/X_train.pkl", "wb"))
pickle.dump(X_test, open("models/X_test.pkl", "wb"))
pickle.dump(y_train, open("models/y_train.pkl", "wb"))
pickle.dump(y_test, open("models/y_test.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("models/feature_names.pkl", "wb"))
print("\nAll files saved successfully")