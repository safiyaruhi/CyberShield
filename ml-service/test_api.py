
import requests

url = "http://127.0.0.1:5000/predict"

test_data = {
    "url": "https://example.com/login"
}

response = requests.post(url, json=test_data)

print("Status code:", response.status_code)
print("API response:")
print(response.json())
