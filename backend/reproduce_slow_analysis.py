import time
from PIL import Image
import numpy as np
import sys
import os

# Add backend to path
sys.path.append(os.getcwd())

from model_manager import ModelManager

def test():
    print("Initializing ModelManager...")
    mm = ModelManager()
    
    # Wait a bit for background threads to start loading (simulating startup)
    print("Waiting 10s for models to start loading...")
    time.sleep(10)
    
    print("Checking loading status:")
    for name, status in mm.loading_status.items():
        print(f"  {name}: {status}")

    # Create dummy image
    img = Image.fromarray(np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8))
    
    print("\nStarting predict_full_suite...")
    start_time = time.time()
    try:
        report = mm.predict_full_suite(img)
        end_time = time.time()
        print(f"\nAnalysis completed in {end_time - start_time:.2f} seconds.")
        
        # Print summary
        if 'ensemble' in report:
            print(f"Consensus: {report['ensemble']['summary']['consensus']}")
            print(f"Total Models Run: {report['ensemble']['summary']['total_models']}")
        
        if 'patches' in report:
            print(f"Patch Conflicts: {report['patches']['conflict_count']}")

    except Exception as e:
        print(f"\nAnalysis crashed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Redirect stdout to file
    with open("reproduce_log.txt", "w") as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        test()

