# model/train_model.py
import pandas as pd
import numpy as np
import pickle
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

# Get the base directory for absolute paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load or create dataset
df = pd.read_csv(os.path.join(base_dir, 'data/triage_sample.csv'))
X = df.drop('triage_level', axis=1)
y = df['triage_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model using joblib instead of pickle for better compatibility
model_path = os.path.join(base_dir, 'model/model.pkl')
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Create and save LIME explainer
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns,
    class_names=[str(i) for i in sorted(y.unique())],
    mode='classification'
)

# Save explainer using joblib
explainer_path = os.path.join(base_dir, 'model/explainer.pkl')
joblib.dump(explainer, explainer_path)
print(f"Explainer saved to {explainer_path}")

# Also save training data for future reference
explainer_data = {
    'training_data': np.array(X_train),
    'feature_names': X.columns,
    'class_names': [str(i) for i in sorted(y.unique())]
}
explainer_data_path = os.path.join(base_dir, 'model/explainer_data.pkl')
joblib.dump(explainer_data, explainer_data_path)
print(f"Explainer data saved to {explainer_data_path}")
