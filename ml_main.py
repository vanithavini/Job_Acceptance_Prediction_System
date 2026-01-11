import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Paths
DATA_PATH = r"C:\Users\Vanitha\OneDrive\Documents\GUVI\capstone_project\job_acceptance_prediction_system\data\cleaned\job_acceptance_f_e_data.csv"
ARTIFACT_PATH = r"C:\Users\Vanitha\OneDrive\Documents\GUVI\capstone_project\job_acceptance_prediction_system\artifacts"

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Initial dataset shape:", df.shape)

# Clean target: status
df["status"] = df["status"].astype(str).str.strip().str.lower()
df["status"] = df["status"].replace({
    "placed": "placed",
    "yes": "placed",
    "accepted": "placed",
    "1": "placed",
    "true": "placed",
    "not placed": "not placed",
    "not_placed": "not placed",
    "rejected": "not placed",
    "no": "not placed",
    "0": "not placed",
    "false": "not placed"
})
df = df[df["status"].isin(["placed", "not placed"])]
df["status"] = df["status"].map({"placed": 1, "not placed": 0})
print("Dataset shape after status cleaning:", df.shape)
print("Target distribution:\n", df["status"].value_counts())

# Safety check
assert df.shape[0] > 0, "Dataset became empty after cleaning!"

# Derived Features
# Interview average
df["interview_score_avg"] = df[["technical_score", "aptitude_score", "communication_score"]].mean(axis=1)

# Placement probability score (based on skills match %)
df["placement_probability_score"] = df["skills_match_percentage"] / 100

# Academic band: 0=low, 1=medium, 2=high
df["academic_band"] = df["degree_percentage"].apply(lambda x: 2 if x >= 75 else 1 if x >= 60 else 0)

# Experience category: 0=junior, 1=mid, 2=senior
df["experience_category"] = df["years_of_experience"].apply(lambda x: 0 if x < 2 else 1 if x < 5 else 2)

# Interview performance: 0=low, 1=medium, 2=high
df["interview_performance"] = df["interview_score_avg"].apply(lambda x: 2 if x >= 75 else 1 if x >= 50 else 0)

# Skills match level: Low/Medium/High
def skills_level(x):
    if x < 50:
        return "Low"
    elif x < 75:
        return "Medium"
    else:
        return "High"

df["skills_match_level"] = df["skills_match_percentage"].apply(skills_level)

# Separate Features & Target
X = df.drop("status", axis=1)
y = df["status"]

# Identify numeric & categorical columns
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Scale numeric features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Save feature column order and numeric columns
feature_columns = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model training
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel trained successfully")
print(f"Accuracy: {accuracy:.4f}")

# Save artifacts
with open(f"{ARTIFACT_PATH}/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(f"{ARTIFACT_PATH}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(f"{ARTIFACT_PATH}/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open(f"{ARTIFACT_PATH}/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

with open(f"{ARTIFACT_PATH}/numeric_columns.pkl", "wb") as f:
    pickle.dump(numeric_cols, f)

print("All artifacts saved successfully. ML pipeline completed!")
