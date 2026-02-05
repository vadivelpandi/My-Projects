
import requests
import time
import sys

def check_server():
    print("Checking server at http://localhost:8000/ ...")
    try:
        r = requests.get("http://localhost:8000/", timeout=5)
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.json()}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    check_server()
