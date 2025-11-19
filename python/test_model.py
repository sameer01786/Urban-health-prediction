import pickle
import pandas as pd

# --- 1. Define Sample Data ---
# These are the demo values from your 'SampleData-1.xlsx'
sample_data = [
    {
        'common_name': 'Yellow Bells',
        'ward_name': '61',  # Note: Must be a string, as in training
        'ownership': 'Private',
        'canopy_dia_m': 1.0,
        'girth_cm': 10.0,
        'height_m': 2.0
    },
    {
        'common_name': 'Fish Tail Palm',
        'ward_name': '8',
        'ownership': 'Private',
        'canopy_dia_m': 4.0,
        'girth_cm': 115.0,
        'height_m': 10.0
    },
    {
        'common_name': 'Subabul',
        'ward_name': '29',
        'ownership': 'On Road',
        'canopy_dia_m': 2.0,
        'girth_cm': 15.0,
        'height_m': 2.0
    },
    {
        # --- Test for missing numerical data (should be imputed) ---
        'common_name': 'Mango',
        'ward_name': '8',
        'ownership': 'Private',
        'canopy_dia_m': None, # Imputer will handle this
        'girth_cm': 13.0,
        'height_m': 2.0
    },
    {
        # --- Test for 'Unknown' categorical data ---
        'common_name': 'A tree not in the training data', # Encoder will handle this
        'ward_name': '999', # Encoder will handle this
        'ownership': 'Unknown', # Imputer will handle this
        'canopy_dia_m': 5.0,
        'girth_cm': 50.0,
        'height_m': 8.0
    }
]

# --- 2. Load the Model ---
pipeline_filename = 'tree_model_pipeline.pkl'
print(f"Loading model pipeline from {pipeline_filename}...")

try:
    with open(pipeline_filename, 'rb') as f:
        model_pipeline = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model file '{pipeline_filename}' not found.")
    print("Please run train_model.py first to create the model file.")
    exit()
except Exception as e:
    print(f"An error occurred loading the model: {e}")
    exit()

print("\n--- Running Predictions ---")

input_df = pd.DataFrame(sample_data)

try:
    predictions = model_pipeline.predict(input_df)
    
    probabilities = model_pipeline.predict_proba(input_df)
    classes = model_pipeline.classes_
    
    for i, prediction in enumerate(predictions):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Input: \n{input_df.iloc[i].to_dict()}")
        print(f"==> Predicted Condition: {prediction}")
        
        print("\nPrediction Probabilities:")
        prob_dict = {classes[j]: f"{probabilities[i][j]*100:.2f}%" for j in range(len(classes))}
        print(prob_dict)

except Exception as e:
    print(f"An error occurred during prediction: {e}")