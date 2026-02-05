
import requests
import numpy as np
from PIL import Image
import io

def create_dummy_image():
    # Create a random RGB image
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf

def test_analyze():
    url = "http://localhost:8001/analyze"
    img_buf = create_dummy_image()
    
    files = {'file': ('test.jpg', img_buf, 'image/jpeg')}
    
    try:
        print("Sending request to backend...")
        response = requests.post(url, files=files)
        response.raise_for_status()
        data = response.json()
        
        print("\nResponse Received:")
        print(f"Total Confidence: {data.get('confidence')}")
        
        forensic = data.get('forensicAnalysis', {})
        print("\nForensic Analysis Results:")
        for k, v in forensic.items():
            print(f"  {k}: {v}")
            
        # Basic validation
        if 'elaScore' not in forensic:
            print("\nFAIL: elaScore missing")
        elif 'forensicScore' not in forensic:
             print("\nFAIL: forensicScore missing")
        else:
             print("\nSUCCESS: Forensic data present.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_analyze()
