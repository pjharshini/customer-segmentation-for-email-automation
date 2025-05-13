from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model, encoder, and scaler
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define column names
binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
categorical_cols = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod"
]
numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        print("Received Data:", data)  # Debugging: Check received input

        # Convert data to DataFrame
        user_df = pd.DataFrame([data])

        # Convert binary categorical columns
        for col in binary_cols:
            if col in user_df:
                user_df[col] = user_df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

        # One-hot encode categorical columns
        user_categorical = encoder.transform(user_df[categorical_cols])

        # Scale numerical columns
        user_numerical = scaler.transform(user_df[numerical_cols])

        # Combine numerical and categorical features
        processed_data = np.hstack((user_numerical, user_categorical))

        # Make prediction
        prediction = model.predict(processed_data)

        # Format response
        result = "Churn" if prediction[0] == 1 else "No Churn"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
