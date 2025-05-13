# Import libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define columns used in preprocessing
binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
categorical_cols = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod"
]
numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

# Function to preprocess input data (used for training and prediction)
def preprocess_input(user_data, encoder, scaler):
    user_df = pd.DataFrame(user_data)

    # Convert binary columns
    for col in binary_cols:
        if col in user_df:
            user_df[col] = user_df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

    # One-hot encode categorical variables
    user_categorical = encoder.transform(user_df[categorical_cols])

    # Scale numerical columns
    user_numerical = scaler.transform(user_df[numerical_cols])

    # Combine numerical and categorical features
    processed_data = np.hstack((user_numerical, user_categorical))

    return processed_data

# Load the dataset
file_path = "Telco-Customer-Churn.xlsx"  # Update this path
df = pd.read_excel(file_path)

# Drop 'customerID' as it is not useful for prediction
df.drop(columns=["customerID"], inplace=True)

# Convert 'TotalCharges' to numeric (handling empty values)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Fill missing 'TotalCharges' values with the column mean
df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)

# Convert binary categorical columns to numerical (1 for Yes/Male, 0 for No/Female)
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

# One-hot encode categorical variables
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  
encoded_categorical = encoder.fit_transform(df[categorical_cols])

# Scale numerical variables
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(df[numerical_cols])

# Combine scaled numerical and encoded categorical features
X = np.hstack((scaled_numerical, encoded_categorical))
y = df["Churn"].values  # Target variable

# Save the encoder & scaler for future use
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12, stratify=y)

# Train Logistic Regression model
model = LogisticRegression(max_iter=5000)  # Increased iterations to avoid convergence warning
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training complete and saved!")
