
from forensic_engine import ForensicEngine
from PIL import Image
import numpy as np
import json

def test_forensic_engine():
    print("Initializing Engine...")
    engine = ForensicEngine()
    
    # Create a dummy image (Random Noise)
    print("Creating test image...")
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    
    print("Running Analysis...")
    results = engine.analyze(img)
    
    print("\n" + "="*50)
    print("RESULTS:")
    print(json.dumps(results, indent=2))
    print("="*50)

    # Check for require keys
    required = ["elaScore", "noisePattern", "frequencyAnalysis", "forensicScore", "details"]
    missing = [k for k in required if k not in results]
    
    if missing:
        print(f"FAILED: Missing keys: {missing}")
    else:
        print("SUCCESS: All keys present.")
        
    # Check detail keys
    detail_keys = ["pixel", "sensor", "frequency", "structural", "fusion"]
    missing_details = [k for k in detail_keys if k not in results['details']]
    if missing_details:
         print(f"FAILED: Missing detail keys: {missing_details}")
    else:
         print("SUCCESS: Detailed structure verified.")

if __name__ == "__main__":
    test_forensic_engine()
