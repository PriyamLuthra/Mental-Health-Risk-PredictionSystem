from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import pandas as pd

# ── App Setup ────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# ── Load Model eagerly at startup ────────────────────────────────
model  = None
scaler = None

def load_objects():
    global model, scaler
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("[OK] Model and Scaler loaded successfully.")
    except FileNotFoundError as e:
        print(f"[ERROR] Model file not found: {e}. Run train_model.py first.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")

# Load on startup
load_objects()

# ── Routes ───────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please run backend/train_model.py first.'
        }), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': 'No JSON body received.'}), 400

    try:
        # ── Feature extraction ───────────────────────────────────
        # Numeric features
        age     = float(data.get('age', 25))
        sleep   = float(data.get('sleep', 7))
        outside = float(data.get('outside', 2))

        # Binary features — accept int 0/1, bool, or string 'Yes'/'No'
        def to_binary(val):
            if isinstance(val, bool):
                return int(val)
            if isinstance(val, (int, float)):
                return 1 if val else 0
            if isinstance(val, str):
                return 1 if val.lower() in ('yes', '1', 'true') else 0
            return 0

        screen   = to_binary(data.get('screen', 0))
        physical = to_binary(data.get('physical', 0))
        diet     = to_binary(data.get('diet', 0))

        # Order must match training: [age_yrs, sueno, horasfuera, ordenador, actividadfisica, dieta]
        feature_names = ['age_yrs', 'sueno', 'horasfuera', 'ordenador', 'actividadfisica', 'dieta']
        features      = pd.DataFrame([[age, sleep, outside, screen, physical, diet]], columns=feature_names)
        features_scaled = scaler.transform(features)

        # ── Prediction with lower threshold for better risk sensitivity ───
        probability = float(model.predict_proba(features_scaled)[0][1]) if hasattr(model, 'predict_proba') else 0.5
        RISK_THRESHOLD = 0.40  # Lower than default 0.5 — catches more true risk cases
        prediction = 1 if probability >= RISK_THRESHOLD else 0

        risk_status = 'High Risk Identified' if prediction == 1 else 'No High Risk Detected'

        response_data = {
            'success'     : True,
            'prediction'  : prediction,
            'risk_status' : risk_status,
            'probability' : round(probability * 100, 2)
        }
        print(f"[PREDICT] input={data}  ->  {risk_status} ({response_data['probability']}%)")
        return jsonify(response_data)

    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Entry Point ───────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
