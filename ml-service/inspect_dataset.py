
import pandas as pd

# Load dataset
data = pd.read_csv("dataset/phishing_dataset.csv")

# Basic dataset information
print("Dataset shape:", data.shape)

print("\nLabel values:")
print(data["label"].unique())

print("\nNumber of records per label:")
print(data["label"].value_counts())

print("\nMissing values:")
print(data.isnull().sum().sum())

print("\nSample URLs:")
print(data[["URL", "Domain", "label"]].head(10))


# 1. Check exact duplicate URLs
print("\n--- DUPLICATE URL CHECK ---")

duplicate_urls = data["URL"].duplicated().sum()
print("Exact duplicate URLs:", duplicate_urls)

# 2. Check duplicate domains
print("\n--- DUPLICATE DOMAIN CHECK ---")

duplicate_domains = data["Domain"].duplicated().sum()
unique_domains = data["Domain"].nunique()

print("Repeated domain records:", duplicate_domains)
print("Unique domains:", unique_domains)

# 3. Check whether the same URL has conflicting labels
print("\n--- URL LABEL CONSISTENCY CHECK ---")

url_label_counts = data.groupby("URL")["label"].nunique()
conflicting_urls = (url_label_counts > 1).sum()

print("URLs with conflicting labels:", conflicting_urls)

# 4. Check domains associated with both labels
print("\n--- DOMAIN LABEL CHECK ---")

domain_label_counts = data.groupby("Domain")["label"].nunique()
mixed_label_domains = (domain_label_counts > 1).sum()

print("Domains appearing with both labels:", mixed_label_domains)

print("\nDataset inspection completed!")
