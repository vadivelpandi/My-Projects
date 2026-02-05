import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
import traceback

model_name = "peterwang512/CNNDetection"

print(f"Attempting to load {model_name}...")

try:
    print("Trying AutoModelForImageClassification...")
    model = AutoModelForImageClassification.from_pretrained(model_name, trust_remote_code=True)
    print("Model loaded.")
    
    print("Trying AutoImageProcessor...")
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    print("Processor loaded.")
    
    print("Success! Model and processor are compatible with AutoClasses.")
except Exception as e:
    print(f"Failed loading with AutoClasses: {e}")
    traceback.print_exc()

# If that fails, it might be a pure PyTorch Hub model or require specific code. 
# Checking if it works with pipeline again but with more timeout patience...
