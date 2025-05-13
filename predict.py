import pickle
import pandas as pd
import numpy as np
from train_model import preprocess_input, binary_cols, categorical_cols, numerical_cols

# Load encoder, scaler, and trained model
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Function to get user input
def get_user_input():
    print("\nEnter customer details to predict churn:\n")

    user_data = {
        "gender": [input("Gender (Male/Female): ").strip().capitalize()],
        "SeniorCitizen": [int(input("Senior Citizen (1 for Yes, 0 for No): "))],
        "Partner": [input("Partner (Yes/No): ").strip().capitalize()],
        "Dependents": [input("Dependents (Yes/No): ").strip().capitalize()],
        "tenure": [int(input("Tenure (months): "))],
        "PhoneService": [input("Phone Service (Yes/No): ").strip().capitalize()],
        "MultipleLines": [input("Multiple Lines (No phone service/No/Yes): ").strip().capitalize()],
        "InternetService": [input("Internet Service (DSL/Fiber optic/No): ").strip().capitalize()],
        "OnlineSecurity": [input("Online Security (No internet service/No/Yes): ").strip().capitalize()],
        "OnlineBackup": [input("Online Backup (No internet service/No/Yes): ").strip().capitalize()],
        "DeviceProtection": [input("Device Protection (No internet service/No/Yes): ").strip().capitalize()],
        "TechSupport": [input("Tech Support (No internet service/No/Yes): ").strip().capitalize()],
        "StreamingTV": [input("Streaming TV (No internet service/No/Yes): ").strip().capitalize()],
        "StreamingMovies": [input("Streaming Movies (No internet service/No/Yes): ").strip().capitalize()],
        "Contract": [input("Contract (Month-to-month/One year/Two year): ").strip().capitalize()],
        "PaperlessBilling": [input("Paperless Billing (Yes/No): ").strip().capitalize()],
        "PaymentMethod": [input("Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)): ").strip()],
        "MonthlyCharges": [float(input("Monthly Charges: "))],
        "TotalCharges": [float(input("Total Charges: "))]
    }

    return user_data 

# Get user input
user_data = get_user_input()

# Convert user input to DataFrame and preprocess
user_df = pd.DataFrame(user_data)

# Ensure columns match training data format
for col in binary_cols:
    if col in user_df:
        user_df[col] = user_df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

# Check if categorical columns exist in user data (fixes missing keys error)
missing_cols = set(categorical_cols) - set(user_df.columns)
for col in missing_cols:
    user_df[col] = "Unknown"  # Add default value

# Preprocess user data
processed_data = preprocess_input(user_df, encoder, scaler)

# Predict churn
prediction = model.predict(processed_data)

# Display result
if prediction[0] == 1:
    print("\n🔴 The customer is **likely to CHURN**.")
else:
    print("\n🟢 The customer is **likely to STAY**.")
