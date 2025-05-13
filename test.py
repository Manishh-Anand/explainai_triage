import joblib
import os
import traceback

# Check if the explainer data file exists
explainer_data_path = "model/explainer_data.pkl"
if os.path.exists(explainer_data_path):
    print(f"Explainer data file found at {explainer_data_path}")
else:
    print(f"Explainer data file not found at {explainer_data_path}")

try:
    # Load the explainer data
    explainer_data = joblib.load(explainer_data_path)

    # Reconstruct the LIME explainer
    from lime.lime_tabular import LimeTabularExplainer
    explainer = LimeTabularExplainer(
        training_data=explainer_data['training_data'],
        feature_names=explainer_data['feature_names'],
        class_names=explainer_data['class_names'],
        mode="classification"
    )

    print("Explainer loaded and reconstructed successfully!")
except Exception as e:
    print("Error loading explainer:")
    traceback.print_exc()  # Prints the detailed error traceback
