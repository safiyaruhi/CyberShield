
import re
import ipaddress
from urllib.parse import urlparse

FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "is_ip",
    "https",
    "subdomain_count",
    "digit_count",
    "letter_count",
    "special_char_count",
    "at_symbol",
    "hyphen_count",
    "dot_count",
    "equals_count",
    "question_count",
    "ampersand_count",
    "percent_count",
    "suspicious_word_count",
    "has_port"
]


def extract_features(url):
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Please enter a valid URL.")

    url = url.strip()

    if len(url) > 2048:
        raise ValueError("URL is too long.")

    if any(char.isspace() or ord(char) < 32 for char in url):
        raise ValueError("URL must not contain spaces or control characters.")

    # Add a scheme if it is missing
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url

    try:
        parsed = urlparse(url)

        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError("Only HTTP and HTTPS URLs are supported.")

        domain = parsed.hostname or ""

        if not domain:
            raise ValueError("Invalid URL: domain not found.")

        # Validate the port number
        port = parsed.port

    except ValueError as error:
        raise ValueError("Invalid URL: " + str(error))

    # Check whether the domain is an IP address
    try:
        ipaddress.ip_address(domain)
        is_ip = 1

    except ValueError:
        is_ip = 0

        # Validate domain name format
        try:
            ascii_domain = domain.rstrip(".").encode("idna").decode("ascii")
        except UnicodeError:
            raise ValueError("Invalid domain name.")

        if len(ascii_domain) > 253:
            raise ValueError("Domain name is too long.")

        labels = ascii_domain.split(".")

# Require a public-style domain with at least two labels
        if len(labels) < 2:
            raise ValueError("Please enter a valid domain name, such as example.com.")

        for label in labels:
            if (
                not label
                or len(label) > 63
                or not re.match(
                    r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$",
                    label
                )
            ):
                raise ValueError("Invalid domain name.")

    suspicious_words = [
        "login", "verify", "update", "secure",
        "account", "banking", "signin", "password",
        "confirm", "wallet"
    ]

    suspicious_count = sum(
        word in url.lower() for word in suspicious_words
    )

    features = {
        "url_length": len(url),
        "domain_length": len(domain),
        "is_ip": is_ip,
        "https": int(parsed.scheme.lower() == "https"),
        "subdomain_count": max(0, len(domain.split(".")) - 2),
        "digit_count": sum(c.isdigit() for c in url),
        "letter_count": sum(c.isalpha() for c in url),
        "special_char_count": sum(not c.isalnum() for c in url),
        "at_symbol": int("@" in url),
        "hyphen_count": url.count("-"),
        "dot_count": url.count("."),
        "equals_count": url.count("="),
        "question_count": url.count("?"),
        "ampersand_count": url.count("&"),
        "percent_count": url.count("%"),
        "suspicious_word_count": suspicious_count,
        "has_port": int(port is not None)
    }

    return features
