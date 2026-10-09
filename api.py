from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import sys

# Ensure root directory is in system path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.database import log_transaction, init_db

app = Flask(__name__)

# Ensure database is ready
init_db()

# Correct paths to models folder from root
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'model.joblib')
SCALER_PATH = os.path.join(ROOT_DIR, 'models', 'scaler.joblib')

print("Loading model and scaler for API...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Create DataFrame from input
        df_new = pd.DataFrame([data])
        
        # Ensure all columns expected by the model are present (fill missing ones with 0.0)
        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in df_new.columns:
                    df_new[col] = 0.0
            # Reorder columns to match model's training order exactly
            df_new = df_new[model.feature_names_in_]
        
        # Scale Time and Amount if present in columns
        cols_to_scale = [c for c in ['Time', 'Amount'] if c in df_new.columns]
        if cols_to_scale:
            df_new[cols_to_scale] = scaler.transform(df_new[cols_to_scale])
        
        # Predict probability of fraud
        fraud_prob = float(model.predict_proba(df_new)[0, 1])
        
        # Lower decision threshold to 0.2 to effectively catch fraud patterns in imbalanced data
        prediction = 1 if fraud_prob >= 0.2 else 0
        
        # Log to SQLite database
        log_transaction(
            time_feature=float(data.get('Time', 0.0)),
            amount=float(data.get('Amount', 0.0)),
            fraud_probability=fraud_prob,
            prediction=prediction
        )
        
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'fraud_probability': fraud_prob,
            'risk_level': 'HIGH' if prediction == 1 else 'LOW'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)