
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

print("Attempting to import main...")
try:
    import main
    print("Successfully imported main.")
except Exception as e:
    print(f"FAILED to import main: {e}")
    import traceback
    traceback.print_exc()

print("Attempting to import forensic_engine...")
try:
    import forensic_engine
    print("Successfully imported forensic_engine.")
    
    # Try a quick test
    try:
        from PIL import Image
        import numpy as np
        fe = forensic_engine.ForensicEngine()
        img = Image.fromarray(np.zeros((100,100,3), dtype=np.uint8))
        res = fe.analyze(img)
        print(f"Forensic engine test result: {res}")
    except Exception as e:
        print(f"Forensic engine runtime test failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"FAILED to import forensic_engine: {e}")
    import traceback
    traceback.print_exc()
