from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
from datetime import datetime
import os
from email_utils import send_email
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('test_model.joblib')

def get_customer_rfm(df, customer_id):
    """Calculate RFM metrics for a specific customer"""
    # Convert InvoiceDate to datetime if it's not already
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Calculate the reference date (current date)
    today = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Filter data for the specific customer
    customer_data = df[df['CustomerID'].astype(str) == str(customer_id)]
    
    if customer_data.empty:
        return None
    
    # Calculate RFM metrics for this customer
    recency = (today - customer_data['InvoiceDate'].max()).days
    # Count total number of transactions (not just unique invoices)
    frequency = len(customer_data)
    monetary = customer_data['TotalPrice'].sum()
    
    return pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'Monetary': [monetary]
    })

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Print the received JSON payload
        print("\nReceived JSON Payload:")
        print(data)
        
        # Parse InvoiceDate from custom format MM/DD/YYYY HH:MM (24hr)
        if 'InvoiceDate' in data:
            data['InvoiceDate'] = datetime.strptime(data['InvoiceDate'], '%m/%d/%Y %H:%M')
        
        # Convert numeric fields to appropriate types
        if 'Quantity' in data:
            data['Quantity'] = int(data['Quantity'])
        if 'UnitPrice' in data:
            data['UnitPrice'] = float(data['UnitPrice'])
            
        # Create TotalPrice field
        data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
        
        # Create DataFrame from the input data
        df = pd.DataFrame([data])
        
        # Try to read the existing Excel file
        try:
            if os.path.exists('data/Database.xlsx'):
                existing_df = pd.read_excel('data/Database.xlsx')
                # Ensure InvoiceDate is datetime
                existing_df['InvoiceDate'] = pd.to_datetime(existing_df['InvoiceDate'], errors='coerce')
                # Convert CustomerID to string for consistent comparison
                existing_df['CustomerID'] = existing_df['CustomerID'].astype(str)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
            else:
                updated_df = df
                
            # Ensure InvoiceDate is datetime and format it
            updated_df['InvoiceDate'] = pd.to_datetime(updated_df['InvoiceDate'], errors='coerce')
            updated_df['InvoiceDate'] = updated_df['InvoiceDate'].dt.strftime('%Y-%m-%d %H:%M:%S')
            # Save the updated dataframe
            updated_df.to_excel('data/Database.xlsx', index=False)
        except Exception as e:
            # If there's an error with the file, just use the new data
            print(f"Error reading/writing Excel file: {e}")
            updated_df = df
            
        # Get customer RFM metrics
        customer_id = str(data['CustomerID'])
        customer_rfm = get_customer_rfm(updated_df, customer_id)
        
        if customer_rfm is None:
            return jsonify({'error': 'Customer not found in database'}), 404
            
        # Get RFM values for the customer
        recency = customer_rfm['Recency'].iloc[0]
        frequency = customer_rfm['Frequency'].iloc[0]
        monetary = customer_rfm['Monetary'].iloc[0]
        
        # Print debug information
        print(f"\nRFM Calculation Details for Customer {customer_id}:")
        print(f"Total transactions: {frequency}")
        print(f"Total spending: {monetary:.2f}")
        print(f"Days since last purchase: {recency}")
        
        # Predict the segment using the saved model
        segment_number = int(model.predict(customer_rfm)[0])  # Ensure it's an integer
        print(f"\nPredicted segment number: {segment_number}")
        
        # Map segment number to template filename and subject
        segment_templates = {
            0: ("active.txt", "Keep the Good Times Rolling – Your Next Treat Is Here!"),
            1: ("at_risk.txt", "We Miss You – Let's Catch Up?"),
            2: ("inactive.txt", "A Fresh Start – On Us!"),
            3: ("loyal.txt", "A Special Thank You – Just for You, Our VIP!"),
            4: ("upcoming.txt", "Welcome! A Surprise Awaits on Your First Visit")
        }
        template_file, subject = segment_templates.get(segment_number, ("inactive.txt", "A Fresh Start – On Us!"))
        template_path = os.path.join("templates", template_file)

        # Read the template and format with RFM values
        try:
            with open(template_path, "r") as f:
                body = f.read().format(
                    CustomerID=customer_id,
                    recency=recency,
                    frequency=frequency,
                    monetary=monetary
                )
        except Exception as e:
            print(f"Error reading template file: {e}")
            body = f"Hi {customer_id},\n\nYour customer segment number is: {segment_number}\n\nYour RFM metrics:\n- Recency: {recency} days since last purchase\n- Frequency: {frequency} total transactions\n- Monetary: {monetary:.2f} total spending\n\nThank you for your business!"

        email = f"{customer_id}@gmail.com"

        try:
            send_email(email, subject, body)
        except Exception as e:
            print(f"Error sending email: {e}")
        
        # Return the segment and RFM values
        return jsonify({
            'segment_number': segment_number,
            'rfm_values': {
                'recency': int(recency),
                'frequency': int(frequency),
                'monetary': float(monetary)
            }
        })
    
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        # Convert JSON data to DataFrame
        df = pd.DataFrame(request.json)
        
        # Get unique customers
        customers = df['CustomerID'].unique()
        
        # Calculate RFM for each customer
        results = []
        for customer_id in customers:
            customer_rfm = get_customer_rfm(df, customer_id)
            if customer_rfm is not None:
                # Predict segment using the model
                segment_number = int(model.predict(customer_rfm)[0])  # Ensure it's an integer
                
                results.append({
                    'CustomerID': customer_id,
                    'Recency': int(customer_rfm['Recency'].iloc[0]),
                    'Frequency': int(customer_rfm['Frequency'].iloc[0]),
                    'Monetary': float(customer_rfm['Monetary'].iloc[0]),
                    'segment_number': segment_number
                })
        
        return jsonify({'rfm_analysis': results})
        
    except Exception as e:
        print(f"Error in analyze_data: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


