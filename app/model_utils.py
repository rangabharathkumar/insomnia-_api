import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load artifacts (model, label encoders, and scaler)
with open('app/artifacts/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('app/artifacts/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('app/artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def preprocess_data(input_data):
    """Preprocess the input data based on the model's expected format."""
    
    # Extract features and apply encoding for categorical variables
    processed_data = input_data.copy()
    
    # Encode categorical variables
    for col in ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure']:
        le = label_encoders.get(col)
        if le:
            processed_data[col] = le.transform([processed_data[col]])[0]

    # Scale numerical variables
    numerical_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 
                      'Physical Activity Level', 'Stress Level', 
                      'Heart Rate', 'Daily Steps']
    
    processed_data[numerical_cols] = scaler.transform([processed_data[numerical_cols]])

    return processed_data

def predict(input_data):
    """Predict the insomnia condition based on the model."""
    processed_data = preprocess_data(input_data)
    
    # Make prediction
    prediction = model.predict([processed_data])
    return prediction[0]
