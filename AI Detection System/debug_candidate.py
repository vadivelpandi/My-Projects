from transformers import pipeline
import traceback

model_name = "dima806/deepfake_vs_real_image_detection"
print(f"Attempting to load {model_name}...")

try:
    pipe = pipeline("image-classification", model=model_name)
    print("Success! Model loaded.")
    print("Labels:", pipe.model.config.id2label)
except Exception as e:
    print(f"Failed loading {model_name}: {e}")
    traceback.print_exc()
