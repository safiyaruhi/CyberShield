
from feature_extraction import extract_features

test_urls = [
    "https://example.com",
    "http://google.com",
    "example.com/login",
    "",
    "hello",
    "ftp://example.com",
    "https://example.com:99999"
]

for url in test_urls:
    print("\nTesting:", repr(url))

    try:
        features = extract_features(url)
        print("Accepted")
        print("Domain length:", features["domain_length"])

    except ValueError as error:
        print("Rejected:", error)
