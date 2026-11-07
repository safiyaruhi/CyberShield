
from feature_extraction import extract_features

test_url = "https://example.com/login"

features = extract_features(test_url)

print("Testing URL:", test_url)
print("\nExtracted features:")

for name, value in features.items():
    print(name, ":", value)

print("\nFeature extraction successful!")
