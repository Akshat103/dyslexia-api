from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained XGBoost model
model = joblib.load("xgboost_model.pkl")

# Define a function to preprocess the input data
def preprocess_input(input_data):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])
    
    # Convert categorical variables to binary (0 or 1)
    input_df['Nativelang'] = input_df['Nativelang'].map({'no': 0, 'yes': 1})
    input_df['Gender'] = input_df['Gender'].map({'male': 0, 'female': 1})
    
    # Binarize the scores
    for i in range(1, 33):
        col_name = f'Score{i}'
        input_df[col_name] = (input_df[col_name] > 0).astype(int)
    
    return input_df

# Define a route for the "Welcome" message
@app.route('/')
def hello_world():
    return 'Welcome at dyslexia predict'

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.json

    # Preprocess the input data
    input_df = preprocess_input(input_data)

    # Make predictions
    prediction = model.predict(input_df)[0]

    # Map the prediction to 'no' or 'yes'
    dyslexia_prediction = 'Yes' if prediction == 1 else 'No'
    
    # Return the prediction
    return jsonify({'Dyslexia': dyslexia_prediction})

if __name__ == '__main__':
    app.run()
