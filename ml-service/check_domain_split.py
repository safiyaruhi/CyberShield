
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Load dataset
data = pd.read_csv("dataset/phishing_dataset.csv")

# Remove exact duplicate URLs
data["URL"] = data["URL"].astype(str).str.strip().str.lower()
data["Domain"] = data["Domain"].astype(str).str.strip().str.lower()

data = data.drop_duplicates(subset=["URL"]).reset_index(drop=True)

X = data["URL"]
y = data["label"].astype(int)
groups = data["Domain"]

# First split: approximately 80% remaining, 20% test
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_val_idx, test_idx = next(
    splitter.split(X, y, groups)
)

# Second split: separate validation from training
X_train_val = X.iloc[train_val_idx]
y_train_val = y.iloc[train_val_idx]
groups_train_val = groups.iloc[train_val_idx]

splitter2 = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=43
)

train_relative_idx, validation_relative_idx = next(
    splitter2.split(
        X_train_val,
        y_train_val,
        groups_train_val
    )
)

train_idx = train_val_idx[train_relative_idx]
validation_idx = train_val_idx[validation_relative_idx]

# Get domain sets
train_domains = set(groups.iloc[train_idx])
validation_domains = set(groups.iloc[validation_idx])
test_domains = set(groups.iloc[test_idx])

print("\n--- DOMAIN-SEPARATED SPLIT ---")
print("Total records after duplicate removal:", len(data))
print("Training records:", len(train_idx))
print("Validation records:", len(validation_idx))
print("Testing records:", len(test_idx))

print("\n--- CLASS DISTRIBUTION ---")

for name, indices in [
    ("Training", train_idx),
    ("Validation", validation_idx),
    ("Testing", test_idx)
]:
    labels = y.iloc[indices]
    print(f"\n{name}:")
    print("Legitimate:", (labels == 1).sum())
    print("Phishing:", (labels == 0).sum())
    print("Phishing percentage:", round((labels == 0).mean() * 100, 2), "%")

print("\n--- DOMAIN OVERLAP CHECK ---")
print("Training-validation overlap:", len(train_domains & validation_domains))
print("Training-testing overlap:", len(train_domains & test_domains))
print("Validation-testing overlap:", len(validation_domains & test_domains))

print("\nDomain-separated split completed!")
