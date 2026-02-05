from transformers import pipeline
import traceback

model_name = "peterwang512/CNNDetection"

print(f"Attempting to load {model_name}...")
try:
    # Try standard load
    pipe = pipeline("image-classification", model=model_name)
    print("Success!")
except Exception as e:
    print(f"Failed with standard pipeline: {e}")
    traceback.print_exc()
    
    print("\nAttempting with trust_remote_code=True...")
    try:
        pipe = pipeline("image-classification", model=model_name, trust_remote_code=True)
        print("Success with trust_remote_code=True!")
    except Exception as e2:
        print(f"Failed with trust_remote_code=True: {e2}")
        traceback.print_exc()
