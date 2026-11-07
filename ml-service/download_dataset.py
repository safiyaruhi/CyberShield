from ucimlrepo import fetch_ucirepo
import pandas as pd

print("Downloading phishing dataset...")

dataset = fetch_ucirepo(id=967)

data = pd.concat(
    [dataset.data.features, dataset.data.targets],
    axis=1
)

print("Dataset shape:", data.shape)
print("\nColumns:")
print(data.columns.tolist())

data.to_csv("dataset/phishing_dataset.csv", index=False)

print("\nDataset saved successfully!")