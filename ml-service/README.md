# CyberShield – Machine Learning Service

## Overview
The CyberShield ML service detects potentially phishing URLs using a Machine Learning model. It extracts URL-based features and predicts whether a submitted URL is Phishing or Legitimate.

## Technology Stack
- Python 3.11
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Setup Instructions

1. Create and activate the Conda environment:

   ```bash
   conda create -n cybershield python=3.11
   conda activate cybershield
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the ML API:

   ```bash
   python app.py
   ```

The service runs at:

`http://127.0.0.1:5000`

## API Endpoints

### 1. Health Check

**Method:** GET

**Endpoint:** `/health`

Example response:

```json
{
  "status": "success",
  "message": "CyberShield ML service is running"
}
```

### 2. URL Prediction

**Method:** POST

**Endpoint:** `/predict`

Request body:

```json
{
  "url": "https://example.com"
}
```

Example response:

```json
{
  "prediction": "Legitimate",
  "phishing_probability": 0.02,
  "risk_score": 2,
  "risk_level": "Low",
  "reasons": [
    "No warning indicators were detected by the basic URL checks. This does not guarantee that the website is safe."
  ]
}
```

**Note:** The response above is an illustrative example, not a guaranteed prediction.

## Response Fields

- `prediction`: Predicted class (Phishing or Legitimate).
- `phishing_probability`: Estimated probability of phishing.
- `risk_score`: Estimated risk score from 0 to 100.
- `risk_level`: Low, Medium, or High.
- `reasons`: Basic URL warning indicators.

## Error Handling

- HTTP 400: Invalid input or unsupported URL.
- HTTP 500: Prediction could not be completed.

## Important Notes

- The service analyzes URL-based features.
- The risk score is an estimate, not a guarantee.
- A low-risk result does not guarantee that a website is safe.
- The service does not visit or open the submitted website.
