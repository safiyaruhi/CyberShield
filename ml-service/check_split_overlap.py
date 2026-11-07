
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("dataset/phishing_dataset.csv")

y = data["label"].astype(int)

# Normalize URL and domain text for comparison
urls = data["URL"].astype(str).str.strip().str.lower()
domains = data["Domain"].astype(str).str.strip().str.lower()

# Reproduce the exact original 70/15/15 split
indices = np.arange(len(data))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=y
)

validation_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=y.iloc[temp_idx]
)

# The final model is trained on training + validation data
final_train_idx = np.concatenate([train_idx, validation_idx])

train_urls = set(urls.iloc[final_train_idx])
test_urls = set(urls.iloc[test_idx])

train_domains = set(domains.iloc[final_train_idx])
test_domains = set(domains.iloc[test_idx])

# Find overlap
shared_urls = train_urls.intersection(test_urls)
shared_domains = train_domains.intersection(test_domains)

# Count test records whose URL/domain also occurs in training
test_url_overlap_records = urls.iloc[test_idx].isin(train_urls).sum()
test_domain_overlap_records = domains.iloc[test_idx].isin(train_domains).sum()

print("\n--- ORIGINAL SPLIT SIZES ---")
print("Training records:", len(train_idx))
print("Validation records:", len(validation_idx))
print("Test records:", len(test_idx))
print("Final training records:", len(final_train_idx))

print("\n--- URL OVERLAP ---")
print("Unique URLs shared between final training and test:", len(shared_urls))
print("Test records with a URL also in training:", test_url_overlap_records)

print("\n--- DOMAIN OVERLAP ---")
print("Unique domains shared between final training and test:", len(shared_domains))
print("Test records with a domain also in training:", test_domain_overlap_records)

print("\nOverlap check completed!")
