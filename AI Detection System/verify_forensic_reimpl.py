
import requests
from PIL import Image
import io
import json

def test_forensic_analysis():
    # Create simple image
    img = Image.new('RGB', (100, 100), color = 'red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    
    print("Sending request to backend...")
    try:
        resp = requests.post("http://localhost:8000/analyze", files={'file': ('test_forensic.jpg', buf, 'image/jpeg')})
        
        if resp.status_code == 200:
            data = resp.json()
            print("Response Keys:", list(data.keys()))
            
            if 'forensicAnalysis' in data:
                fa = data['forensicAnalysis']
                print("Forensic Analysis Found:", json.dumps(fa, indent=2))
                
                if 'elaScore' in fa and 'noisePattern' in fa:
                    print("PASS: Forensic keys present.")
                else:
                    print("FAIL: Missing specific forensic keys.")
            else:
                print("FAIL: forensicAnalysis key missing.")
                
            # Check model count
            summary = data.get('modelConsensus', {})
            total_models = summary.get('totalModels')
            print(f"Total Models used: {total_models}")
            if total_models == 8:
                 print("PASS: Correct model count (8).")
            else:
                 print(f"FAIL: Expected 8 models, got {total_models}")

            # Check Combined Logic
            print("-" * 20)
            print("Verifying Combined Logic:")
            pred = data.get('prediction')
            conf = data.get('confidence')
            print(f"Final Prediction: {pred}")
            print(f"Final Confidence: {conf}")

            if pred in ["AI Generated", "Real"] and isinstance(conf, (int, float)):
                print("PASS: Combined fields present and valid.")
            else:
                print("FAIL: Combined fields missing or invalid.")


        else:
            print(f"FAIL: Server error {resp.status_code}")
            try:
                print(resp.json())
            except:
                print(resp.text)
            
    except Exception as e:
        print(f"FAIL: Connection error: {e}")

if __name__ == "__main__":
    test_forensic_analysis()
