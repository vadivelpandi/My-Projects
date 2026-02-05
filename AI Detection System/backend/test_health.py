import requests
import time

print("Testing backend health...")
try:
    response = requests.get("http://127.0.0.1:8000/", timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except requests.exceptions.Timeout:
    print("ERROR: Request timed out - backend is not responding")
except requests.exceptions.ConnectionError:
    print("ERROR: Cannot connect to backend")
except Exception as e:
    print(f"ERROR: {e}")
