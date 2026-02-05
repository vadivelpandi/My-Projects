from transformers import pipeline
from PIL import Image
import requests

# Load model
model_name = "ash12321/flux-detector-final"
print(f"Loading {model_name}...")
pipe = pipeline("image-classification", model=model_name)

# Create a dummy image (or use a real one if available, but black square is fine for label check)
image = Image.new('RGB', (224, 224), color='red')

print("Running prediction on dummy image...")
preds = pipe(image)

print("Raw Predictions:")
for p in preds:
    print(p)
