
import os
import joblib
import pandas as pd
import numpy as np

from feature_extraction import extract_features, FEATURE_NAMES

from sklearn.model_selection import StratifiedGroupKFold
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

# 1. Load and clean dataset
print("Loading dataset...")

data = pd.read_csv("dataset/phishing_dataset.csv")

data["URL"] = data["URL"].astype(str).str.strip().str.lower()
data["Domain"] = data["Domain"].astype(str).str.strip().str.lower()

data = data.drop_duplicates(subset=["URL"]).reset_index(drop=True)

X_urls = data["URL"]
y = data["label"].astype(int)
groups = data["Domain"]

print("Dataset records:", len(data))

# 2. Recreate domain-separated splits
print("\nCreating domain-separated splits...")

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_val_idx, test_idx = next(
    splitter.split(X_urls, y, groups)
)

splitter2 = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=43
)

train_relative_idx, validation_relative_idx = next(
    splitter2.split(
        X_urls.iloc[train_val_idx],
        y.iloc[train_val_idx],
        groups.iloc[train_val_idx]
    )
)

train_idx = train_val_idx[train_relative_idx]
validation_idx = train_val_idx[validation_relative_idx]

# 3. Extract URL features
print("\nExtracting URL features...")

feature_rows = []

for i, url in enumerate(X_urls):
    feature_rows.append(extract_features(url))

    if (i + 1) % 20000 == 0:
        print("Processed:", i + 1)

X = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)

print("Feature extraction completed:", X.shape)

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]

X_validation = X.iloc[validation_idx]
y_validation = y.iloc[validation_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# 4. Define models
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

# 5. Compare models on validation data
best_model = None
best_model_name = None
best_f1 = -1

print("\nComparing models...")

for name, model in models.items():
    print("\nTraining:", name)

    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_validation,
        predictions,
        labels=[0, 1],
        zero_division=0
    )

    print("Validation accuracy:", accuracy_score(
        y_validation, predictions
    ))
    print("Phishing precision:", precision[0])
    print("Phishing recall:", recall[0])
    print("Phishing F1-score:", f1[0])

    if f1[0] > best_f1:
        best_f1 = f1[0]
        best_model = model
        best_model_name = name

# 6. Retrain selected model on training + validation
print("\nSelected model:", best_model_name)

train_validation_idx = np.concatenate([
    train_idx,
    validation_idx
])

best_model.fit(
    X.iloc[train_validation_idx],
    y.iloc[train_validation_idx]
)

# 7. Final evaluation on untouched test data
print("\nFinal domain-separated test evaluation")

test_predictions = best_model.predict(X_test)

print("\nAccuracy:", accuracy_score(
    y_test, test_predictions
))

print("\nClassification report:")
print(classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["Phishing", "Legitimate"],
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
))

# 8. Save separately from original model
os.makedirs("models", exist_ok=True)

model_bundle = {
    "model": best_model,
    "model_name": best_model_name,
    "feature_names": FEATURE_NAMES,
    "label_mapping": {
        0: "Phishing",
        1: "Legitimate"
    },
    "evaluation": "domain-separated split"
}

joblib.dump(
    model_bundle,
    "models/phishing_model_domain_split.pkl"
)

print("\nDomain-separated model saved!")
print("Location: models/phishing_model_domain_split.pkl")
