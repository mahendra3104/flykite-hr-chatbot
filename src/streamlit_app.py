import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
# TODO: Replace with your Hugging Face user ID for the model if different from the space owner
model_path = hf_hub_download(repo_id="Mahendra-ML/airline-package-model", filename="best_airline_package_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Airline Package Prediction")
st.write("Fill the customer details below to predict if they'll purchase an airline package")

# Collect user input (Placeholder for actual input fields)
# For demonstration, we'll use a dummy DataFrame


# ----------------------------
# Prepare input data
# ----------------------------
input_data = pd.DataFrame([{
    # Example placeholder values for input features
    # Replace with actual features required by your model
    'feature_1': 0.5,
    'feature_2': 100,
    'feature_3': 'category_A'
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prob = model.predict_proba(input_data)[0,1]
    pred = int(prob >= classification_threshold)
    result = "will purchase the airline package" if pred == 1 else "is unlikely to purchase"
    st.write(f"Prediction: Customer {result}")
