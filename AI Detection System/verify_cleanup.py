
import requests
from PIL import Image
import io
import json

def test_basic_analysis():
    # Create simple image
    img = Image.new('RGB', (100, 100), color = 'red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    
    print("Sending request to backend...")
    try:
        resp = requests.post("http://localhost:8000/analyze", files={'file': ('test.jpg', buf, 'image/jpeg')})
        
        if resp.status_code == 200:
            data = resp.json()
            print("Response Keys:", list(data.keys()))
            
            if 'metadataAnalysis' not in data and 'forensicAnalysis' not in data:
                print("PASS: Cleanup verified. No metadata/forensic keys.")
            else:
                print("FAIL: Metadata or Forensic data still present.")
                if 'metadataAnalysis' in data: print("- Found metadataAnalysis")
                if 'forensicAnalysis' in data: print("- Found forensicAnalysis")
        else:
            print(f"FAIL: Server error {resp.status_code}")
            print(resp.text)
            
    except Exception as e:
        print(f"FAIL: Connection error: {e}")

if __name__ == "__main__":
    test_basic_analysis()
