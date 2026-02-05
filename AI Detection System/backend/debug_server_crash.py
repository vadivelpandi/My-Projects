import requests
import time

def test_analyze():
    url = "http://127.0.0.1:8001/analyze"
    # Create a dummy image
    from PIL import Image
    import io
    import numpy as np
    
    img = Image.fromarray(np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    
    files = {"file": ("test.jpg", buf, "image/jpeg")}
    
    print("Sending request to /analyze...")
    try:
        start = time.time()
        response = requests.post(url, files=files, timeout=60)
        end = time.time()
        print(f"Status Code: {response.status_code}")
        print(f"Time: {end - start:.2f}s")
        if response.status_code == 200:
            print("Success!")
            print(response.json().keys())
        else:
            print("Failed")
            print(response.text)
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_analyze()
