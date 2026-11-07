
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

from feature_extraction import extract_features, FEATURE_NAMES

app = Flask(__name__)
CORS(app)

# Load the trained model once when the API starts
bundle = joblib.load("models/phishing_model.pkl")
model = bundle["model"]

print("Loaded model:", bundle["model_name"])



def generate_reasons(features, prediction):
    reasons = []

    if features["is_ip"] == 1:
        reasons.append("The URL uses an IP address instead of a domain name.")

    if features["at_symbol"] == 1:
        reasons.append("The URL contains an @ symbol, which can obscure the destination.")

    if features["url_length"] > 75:
        reasons.append("The URL is unusually long.")

    if features["subdomain_count"] >= 3:
        reasons.append("The URL contains multiple subdomains.")

    if features["suspicious_word_count"] >= 2:
        reasons.append("The URL contains multiple sensitive or suspicious words.")

    if features["hyphen_count"] >= 3:
        reasons.append("The URL contains several hyphens.")

    if features["https"] == 0:
        reasons.append("The URL does not use HTTPS encryption.")

    if not reasons and prediction == "Phishing":
        reasons.append(
            "The machine learning model detected a suspicious pattern "
            "across the URL features, although no individual warning "
            "rule was triggered."
        )

    elif not reasons:
        reasons.append(
            "No warning indicators were detected by the basic URL checks. "
            "This does not guarantee that the website is safe."
        )

    return reasons


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "CyberShield ML service is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Send a valid JSON object."}), 400

        url = data.get("url")

        if not isinstance(url, str) or not url.strip():
            return jsonify({"error": "Please provide a URL."}), 400

        if len(url) > 2048:
            return jsonify({"error": "URL is too long."}), 400

        # Extract exactly the same features used during training
        features = extract_features(url)

        input_data = pd.DataFrame(
            [[features[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES
        )

        # Predict class: 0 = Phishing, 1 = Legitimate
        prediction = int(model.predict(input_data)[0])

        # Get estimated probability of phishing (class 0)
        probabilities = model.predict_proba(input_data)[0]
        class_index = list(model.classes_).index(0)

        phishing_probability = float(probabilities[class_index])

        risk_score = round(phishing_probability * 100)

        if risk_score >= 70:
            risk_level = "High"
        elif risk_score >= 30:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        result = {
            "prediction": (
                "Phishing" if prediction == 0 else "Legitimate"
            ),
            "phishing_probability": round(phishing_probability, 4),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "reasons": generate_reasons(
    features,
    "Phishing" if prediction == 0 else "Legitimate"
)
        }

        return jsonify(result)

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction could not be completed."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
