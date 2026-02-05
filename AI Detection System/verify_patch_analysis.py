import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from model_manager import ModelManager
from PIL import Image, ImageDraw
import numpy as np

def create_test_image():
    # Create a 512x512 image
    img = Image.new('RGB', (512, 512), color='white')
    draw = ImageDraw.Draw(img)
    # Draw some random rectangles to make it interesting
    draw.rectangle([100, 100, 200, 200], fill='red')
    draw.rectangle([300, 300, 400, 400], fill='blue')
    return img

def mock_run_ensemble(image):
    # Mocking the ensemble to return AI for top-left quadrant and Real for others
    # Logic: if average pixel color is "reddish", say AI, else Real?
    # Or just random for testing structure.
    # Let's mock based on image quadrant to test conflict logic.
    
    # Simple check: pass a mock function to ModelManager? 
    # Python is dynamic, we can overwrite the method on the instance.
    return {
        "detailed_results": [],
        "summary": {
            "total_models": 5,
            "ai_votes": 5,
            "real_votes": 0,
            "average_confidence": 0.95,
            "consensus": "AI Generated"
        }
    }

def test_patch_logic():
    print("Initializing ModelManager...")
    mm = ModelManager()
    
    # Mock the _run_ensemble method to avoid loading heavy models and to control output
    original_run_ensemble = mm._run_ensemble
    
    # Define a mock that varies based on image content (or just random for now if we can't easily map)
    # Actually, let's just let it run 'real' models if they are loaded? 
    # No, they take time. Let's mock.
    
    # We want to simulate a conflict: Global says Real, patches say AI?
    # Or Global AI, patches Real.
    
    print("Mocking _run_ensemble...")
    
    def mock_ensemble_ai(image):
         return {
            "detailed_results": [],
            "summary": {"total_models":5, "ai_votes":5, "real_votes":0, "average_confidence": 99.0, "consensus": "AI Generated"}
        }
    
    def mock_ensemble_real(image):
         return {
            "detailed_results": [],
            "summary": {"total_models":5, "ai_votes":0, "real_votes":5, "average_confidence": 1.0, "consensus": "Real"}
        }

    # Scenario 1: Consistent AI
    # Global = AI, Patches = AI
    print("\n--- Test 1: Consistent AI ---")
    mm._run_ensemble = mock_ensemble_ai
    img = create_test_image()
    res = mm.predict_full_suite(img)
    
    print(f"Global Consensus: {res['ensemble']['summary']['consensus']}")
    print(f"Patch Conflict Detected: {res['patches']['conflict_detected']}")
    print(f"Consistency Level: {res['patches']['consistency_level']}")
    
    if res['patches']['conflict_detected'] == 'No':
        print("PASS: Consistent AI detected as Non-Conflict")
    else:
        print("FAIL: Should be Consistent")

    # Scenario 2: Conflict (Global Real, Local AI Hotspots)
    print("\n--- Test 2: Local AI Conflict ---")
    
    # We need a custom mock that returns AI for patches but Real for Global
    # Since predict_full_suite calls _run_ensemble for global first, then _analyze_patches calls it for patches.
    # We can use a counter or state.
    
    class MockEnsemble:
        def __init__(self):
            self.call_count = 0
            
        def __call__(self, image):
            self.call_count += 1
            if self.call_count <= 3: # Global call, 50% call, 25% call -> First 3 calls
                 return {
                    "detailed_results": [],
                    "summary": {"total_models":5, "ai_votes":0, "real_votes":5, "average_confidence": 10.0, "consensus": "Real"}
                }
            else:
                # Patch calls. 16 patches.
                # Let's make patches 0 and 1 AI (Hotspots)
                # Call count starts at 4. 
                # Patch 0 is call 4. Patch 1 is call 5.
                if self.call_count in [4, 5, 6]: 
                     return {
                        "detailed_results": [],
                        "summary": {"total_models":5, "ai_votes":5, "real_votes":0, "average_confidence": 95.0, "consensus": "AI Generated"}
                    }
                else:
                    return {
                        "detailed_results": [],
                        "summary": {"total_models":5, "ai_votes":0, "real_votes":5, "average_confidence": 10.0, "consensus": "Real"}
                    }
    
    mock = MockEnsemble()
    mm._run_ensemble = mock
    
    res = mm.predict_full_suite(img)
    print(f"Global Consensus: {res['ensemble']['summary']['consensus']}")
    print(f"Patch Scores (first 3): {[p['verdict'] for p in res['patches']['patch_scores'][:3]]}")
    print(f"Patch Conflict Detected: {res['patches']['conflict_detected']}")
    print(f"Suspected Regions: {res['patches']['suspected_regions']}")

    if res['patches']['conflict_detected'] == 'Yes':
        print("PASS: Conflict Successfully Detected")
    else:
        print("FAIL: Conflict NOT Detected")

if __name__ == "__main__":
    test_patch_logic()
