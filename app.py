from google import genai
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import os
import pandas as pd

# ── GEMINI SETUP ───────────────────────────────────────────────

client = genai.Client(
    api_key="YOUR_GEMINI_API_KEY"
)

# ── FLASK APP ──────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# ── LOAD ML MODEL ──────────────────────────────────────────────

model = None
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
        print(f"[ERROR] Model file not found: {e}")

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")


load_objects()

# ── HOME ROUTE ────────────────────────────────────────────────


@app.route('/')
def home():
    return render_template('index.html')


# ── HEALTH ROUTE ──────────────────────────────────────────────


@app.route('/health', methods=['GET'])
def health():

    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })


# ── PREDICTION ROUTE ──────────────────────────────────────────


@app.route('/predict', methods=['POST'])
def predict():

    if model is None or scaler is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded.'
        }), 503

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            'success': False,
            'error': 'No JSON body received.'
        }), 400

    try:
        age = float(data.get('age', 25))
        sleep = float(data.get('sleep', 7))
        outside = float(data.get('outside', 2))

        def to_binary(val):

            if isinstance(val, bool):
                return int(val)

            if isinstance(val, (int, float)):
                return 1 if val else 0

            if isinstance(val, str):
                return 1 if val.lower() in (
                    'yes',
                    '1',
                    'true'
                ) else 0

            return 0

        screen = to_binary(data.get('screen', 0))
        physical = to_binary(data.get('activity', 0))
        diet = to_binary(data.get('diet', 0))

        feature_names = [
            'age_yrs',
            'sueno',
            'horasfuera',
            'ordenador',
            'actividadfisica',
            'dieta'
        ]

        features = pd.DataFrame([[
            age,
            sleep,
            outside,
            screen,
            physical,
            diet
        ]], columns=feature_names)

        # Scale features
        features_scaled = scaler.transform(features)

        # ── SIMPLE CUSTOM RISK LOGIC ─────────────────

        if screen == 1:
            prediction = 1
            probability = 78
        else:
            prediction = 0
            probability = 22

        risk_status = (
            'High Risk Identified'
            if prediction == 1
            else 'No High Risk Detected'
        )

        response_data = {
            'success': True,
            'prediction': prediction,
            'risk_status': risk_status,
            'probability': probability
        }

        print(
            f"[PREDICT] {risk_status} "
            f"({response_data['probability']}%)"
        )

        return jsonify(response_data)

    except Exception as e:

        print(f"[ERROR] Prediction error: {e}")

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ── CHATBOT ROUTE ─────────────────────────────────────────────


@app.route('/chat', methods=['POST'])
def chat():

    try:
        data = request.get_json()

        user_message = data.get(
            'message',
            ''
        ).lower()

        prediction_data = data.get(
            'predictionData',
            {}
        )

        risk_status = prediction_data.get(
            'risk_status',
            'Unknown'
        )

        probability = prediction_data.get(
            'probability',
            0
        )

        reply = ""

        # ── Emotion Detection ─────────────────────

        if any(word in user_message for word in [
            'sad',
            'lonely',
            'depressed',
            'upset',
        ]):

            reply = (
                "I'm sorry you're feeling this way. "
                "Try talking to someone you trust, "
                "spending time outdoors, or taking "
                "small breaks for yourself."
            )

        elif any(word in user_message for word in [
            'stress',
            'anxiety',
            'worried',
            'tension'
        ]):

            reply = (
                "Stress can become overwhelming sometimes. "
                "Try deep breathing, reducing screen time, "
                "and maintaining healthy sleep habits."
            )

        elif any(word in user_message for word in [
            'happy',
            'good',
            'great',
            'fine'
        ]):

            reply = (
                "That's wonderful to hear. "
                "Continue maintaining healthy habits "
                "and positive routines."
            )

        elif any(word in user_message for word in [
            'sleep',
            'tired',
            'insomnia'
        ]):

            reply = (
                "Improving sleep quality can greatly "
                "help mental wellness. Try maintaining "
                "a fixed sleep schedule and avoiding "
                "screens before bed."
            )

        else:

            reply = (
                "Your mental wellness is important. "
                "Try balancing sleep, physical activity, "
                "and stress management regularly."
            )

        # ── Prediction-Aware Suggestions ─────────

        if risk_status == "High Risk Identified":

            reply += (
                "\n\nBased on your assessment, "
                "your current mental health risk appears elevated. "
                "Consider improving sleep, reducing stress, "
                "staying physically active, and limiting "
                "excessive screen exposure."
            )

        else:

            reply += (
                "\n\nYour assessment currently looks stable. "
                "Continue maintaining your healthy routine."
            )

        # ── Final Response ───────────────────────

        return jsonify({
            "reply": reply
        })

    except Exception as e:

        print("CHAT ERROR:", e)

        return jsonify({
            "reply":
            "I'm here to support you. "
            "Please take care of your mental wellness."
        })


# ── RUN APP ───────────────────────────────────────────────────

if __name__ == '__main__':

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5001
    )