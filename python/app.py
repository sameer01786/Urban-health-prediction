import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS 

print("Starting Python ML Server...")

app = Flask(__name__)

CORS(app) 

try:
    with open('tree_model_pipeline.pkl', 'rb') as f:
        model_pipeline = pickle.load(f)
    print("Model pipeline loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model pipeline file not found. Please run train_model.py first.")
    exit()

print("Extracting categories from model pipeline...")
try:

    CATEGORICAL_FEATURES = ['common_name', 'ward_name', 'ownership']
    
    encoder = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
    
    categories_dict = {}
    for i, feature in enumerate(CATEGORICAL_FEATURES):
        categories_list = encoder.categories_[i].tolist()
        
       
            
        categories_dict[feature] = categories_list
        
    print(f"Found {len(categories_dict['common_name'])} common_names, {len(categories_dict['ward_name'])} wards, {len(categories_dict['ownership'])} owners.")

except Exception as e:
    print(f"Error extracting categories: {e}")
    print("Defaulting to empty categories.")
    categories_dict = {
        'common_name': ['Error: Check Logs'],
        'ward_name': ['Error: Check Logs'],
        'ownership': ['Error: Check Logs']
    }


@app.route('/')
def home():
    return "Python ML Server is running. Use /predict and /get-categories."

@app.route('/get-categories', methods=['GET'])
def get_categories():
    """
    Provides the frontend with all unique categories
    for building dynamic dropdowns.
    """
    try:
        return jsonify(categories_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received data for prediction: {data}")

        input_df = pd.DataFrame(data, index=[0])

        prediction = model_pipeline.predict(input_df)

        predicted_value = prediction[0]

        print(f"Prediction: {predicted_value}")
        return jsonify({'health_prediction': predicted_value})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)