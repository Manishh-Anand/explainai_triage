# utils/lime_explainer.py
import pickle
import numpy as np

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)

def explain_instance(instance, feature_names):
    exp = explainer.explain_instance(
        data_row=np.array(instance),
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    return exp
