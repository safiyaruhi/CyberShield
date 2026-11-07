
import os
import joblib
import pandas as pd

from feature_extraction import extract_features, FEATURE_NAMES

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# -----------------------------------
# 1. Load dataset
# -----------------------------------

print("Loading dataset...")

data = pd.read_csv("dataset/phishing_dataset.csv")

# 1 = Legitimate, 0 = Phishing
X_urls = data["URL"].astype(str)
y = data["label"].astype(int)

print("Dataset loaded:", len(data), "records")


# -----------------------------------
# 2. Extract URL features
# -----------------------------------

print("\nExtracting URL features...")

feature_rows = []

for index, url in enumerate(X_urls):
    try:
        features = extract_features(url)
        feature_rows.append(features)

    except Exception as error:
        raise ValueError(
            f"Feature extraction failed at row {index}: {url}"
        ) from error

    if (index + 1) % 20000 == 0:
        print("Processed:", index + 1, "URLs")

X = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)

print("Feature extraction completed!")
print("Feature matrix shape:", X.shape)


# -----------------------------------
# 3. Split dataset
# -----------------------------------

# First: 70% training, 30% remaining
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Split remaining 30% into 15% validation and 15% test
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_validation))
print("Testing records:", len(X_test))


# -----------------------------------
# 4. Define machine learning models
# -----------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    )
}


# -----------------------------------
# 5. Train and compare on validation
# -----------------------------------

best_model = None
best_model_name = None
best_validation_f1 = -1

print("\nComparing models...")

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    # Class 0 represents phishing
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_validation,
        predictions,
        labels=[0, 1],
        zero_division=0
    )

    phishing_f1 = f1[0]

    print("Validation accuracy:", accuracy_score(
        y_validation, predictions
    ))

    print("Phishing precision:", precision[0])
    print("Phishing recall:", recall[0])
    print("Phishing F1-score:", phishing_f1)

    # Select the model with the highest phishing F1-score
    if phishing_f1 > best_validation_f1:
        best_validation_f1 = phishing_f1
        best_model = model
        best_model_name = name


# -----------------------------------
# 6. Retrain selected model
# -----------------------------------

print("\nSelected model:", best_model_name)

X_train_validation = pd.concat(
    [X_train, X_validation]
)

y_train_validation = pd.concat(
    [y_train, y_validation]
)

best_model.fit(X_train_validation, y_train_validation)


# -----------------------------------
# 7. Final testing
# -----------------------------------

print("\nFinal evaluation on unseen test data...")

test_predictions = best_model.predict(X_test)

print("\nAccuracy:", accuracy_score(
    y_test, test_predictions
))

print("\nClassification Report:")

print(classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["Phishing", "Legitimate"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
))


# -----------------------------------
# 8. Save trained model
# -----------------------------------

os.makedirs("models", exist_ok=True)

model_bundle = {
    "model": best_model,
    "model_name": best_model_name,
    "feature_names": FEATURE_NAMES,
    "label_mapping": {
        0: "Phishing",
        1: "Legitimate"
    }
}

joblib.dump(
    model_bundle,
    "models/phishing_model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/phishing_model.pkl")
