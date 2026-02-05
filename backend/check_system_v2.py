
import requests
import cv2
import numpy as np

# Create a dummy image
img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
cv2.imwrite("test_v2.jpg", img)

url = "http://127.0.0.1:8000/analyze"
files = {'file': open('test_v2.jpg', 'rb')}

print("Sending request to System 2.0...")
try:
    response = requests.post(url, files=files)
    if response.status_code == 200:
        data = response.json()
        print("\n--- Response Keys ---")
        print(data.keys())
        

        if 'detailed_steps' in data and 'modelConsensus' in data:
            print("\n--- Detailed Steps Verified ---")
            steps = data['detailed_steps']
            print(f"Step 1 (Ensemble): {'OK' if 'step1_ensemble' in steps else 'MISSING'}")
            print(f"Step 2 (Camera):   {'OK' if 'step2_camera' in steps else 'MISSING'}")
            
            print("\n--- Legacy Keys Verified (for Frontend) ---")
            print(f"modelConsensus: {'OK' if 'modelConsensus' in data else 'MISSING'}")
            print(f"detailedModels: {'OK' if 'detailedModels' in data else 'MISSING'}")
            
            print(f"\nFINAL VERDICT: {data['classification']} ({data['ai_probability']}%)")
        else:
            print("ERROR: Missing 'detailed_steps' OR 'modelConsensus' keys!")
    else:
        print(f"Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Connection Failed: {e}")
    print("Make sure uvicorn is running.")
