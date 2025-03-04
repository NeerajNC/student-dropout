from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Define model path
MODEL_PATH = "student_dropout_model.pkl"

# Load the trained model
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        print("✅ Model loaded successfully!")
    except pickle.UnpicklingError:
        print("❌ Error: Model file is corrupted. Re-save and retry.")
        model = None
else:
    print("❌ Error: Model file not found. Train and save the model first.")
    model = None

# Expected features from the dataset
expected_features = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder",
    "Age at enrollment", "International", "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP"
]

# Home route
@app.route('/')
def home():
    return "🎓 Student Dropout Prediction API is running! 🚀"

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Check logs.'})

    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert input JSON into a DataFrame
        df = pd.DataFrame([data])

        # Ensure correct feature order
        df = df[expected_features]

        # Make prediction
        prediction = model.predict(df)[0]

        # Convert prediction output
        result = "Dropout" if prediction == "Dropout" else "Graduate"

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
