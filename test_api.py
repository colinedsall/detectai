# test_api.py
import requests
import json

# Read the content of your file
file_path = '/Users/colin/Documents/GitHub/cursor_test/detectai/example4.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    text_content = f.read()

# Prepare the JSON payload
payload = {
    "text": text_content
}

# Send the request
url = "http://127.0.0.1:8000/v1/detect/text"
headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, data=json.dumps(payload))

print(f"Status Code: {response.status_code}")
print("Response JSON:")
print(response.json())