
print("Starting forensic check...")
try:
    import forensic_engine
    print("Imported forensic_engine successfully.")
except Exception as e:
    print(f"Import failed: {e}")
    exit(1)

try:
    print("Testing class init...")
    fe = forensic_engine.ForensicEngine()
    print("Class initialized.")
    
    from PIL import Image
    import numpy as np
    
    print("Creating dummy image...")
    # Create random image
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    
    print("Running analyze...")
    res = fe.analyze(img)
    print(f"Result keys: {list(res.keys())}")
    print("Forensic check passed.")
except Exception as e:
    print(f"Runtime error: {e}")
    import traceback
    traceback.print_exc()
