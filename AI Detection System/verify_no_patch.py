import requests
from PIL import Image
import io

# Create a simple red image
img = Image.new('RGB', (200, 200), color='red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_byte_arr.seek(0)

url = 'http://localhost:8000/analyze'
files = {'file': ('test_red.jpg', img_byte_arr, 'image/jpeg')}

try:
    print("Sending request to backend...")
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        data = response.json()
        print("Response received.")
        
        # Check step4_patches
        steps = data.get('detailed_steps', {})
        patches = steps.get('step4_patches')
        
        if patches is None:
            print("SUCCESS: step4_patches is None as expected.")
        else:
            print(f"FAILURE: step4_patches is present: {patches}")
            
        print("Verification complete.")
    else:
        print(f"FAILURE: Backend returned status {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"FAILURE: Could not connect to backend: {e}")
