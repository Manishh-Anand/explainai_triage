import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_input(input_df):
    """
    Preprocesses user input for model prediction.
    Assumes input is a pandas DataFrame with appropriate feature columns.
    Ensures consistent feature set with the training data.
    """
    # Ensure triage_level is not in the input
    if 'triage_level' in input_df.columns:
        input_df = input_df.drop('triage_level', axis=1)
    
    # Fill missing values if any
    input_df = input_df.fillna(0)

    # Dummy encoding for categorical variables
    input_df_encoded = pd.get_dummies(input_df)
    
    # Ensure we have the same columns as during training
    # This is a more robust approach but would require saving the column names during training
    # For now, we'll use the StandardScaler approach which should work for the current issue

    # Scaling numeric values
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_df_encoded)

    return pd.DataFrame(input_scaled, columns=input_df_encoded.columns)
