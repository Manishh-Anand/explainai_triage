import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
import os
import joblib

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Generate synthetic data
np.random.seed(42)
n_samples = 300
data = {
    "Age": np.random.randint(0, 100, n_samples),
    "Heart Rate": np.random.randint(50, 180, n_samples),
    "Systolic BP": np.random.randint(90, 200, n_samples),
    "Pain Level": np.random.randint(0, 6, n_samples),
    "Arrival Mode": np.random.choice([0, 1], n_samples),  # 0 = walk-in, 1 = ambulance
    "Symptom": np.random.choice([0, 1, 2, 3], n_samples),  # 0=Chest Pain, etc.
    "triage_level": np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.4, 0.2])
}
df = pd.DataFrame(data)
df.to_csv("data/triage_sample.csv", index=False)

# Train model
X = df.drop("triage_level", axis=1)
y = df["triage_level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model using joblib
joblib.dump(model, "model/model.pkl")

# LIME explainer
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=[str(i) for i in sorted(y.unique())],
    mode="classification"
)

# Instead of saving the explainer directly, save only essential information
explainer_data = {
    'training_data': np.array(X_train),
    'feature_names': X.columns.tolist(),
    'class_names': [str(i) for i in sorted(y.unique())]
}

# Save explainer essential data
joblib.dump(explainer_data, "model/explainer_data.pkl")

print("✅ Files saved: triage_sample.csv, model.pkl, explainer_data.pkl")
