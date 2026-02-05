from transformers import pipeline

models_to_test = [
    "google/vit-base-patch16-224",
    "NHNDQ/vit-base-patch16-224-finetuned-deepfake",
    "Wvolf/ViT_Deepfake_Detection"
]

print("Starting model load check...")

for model in models_to_test:
    print(f"--- Testing {model} ---")
    try:
        pipe = pipeline("image-classification", model=model)
        print(f"SUCCESS: {model} loaded.")
        # Try a dummy prediction
        # (We don't need a real image, pipeline might complain or we can mock it, but loading is the main step)
        # To be safe, let's not predict, just load.
    except Exception as e:
        print(f"FAILURE: {model} failed to load.")
        print(f"Error: {e}")

print("Finished.")
