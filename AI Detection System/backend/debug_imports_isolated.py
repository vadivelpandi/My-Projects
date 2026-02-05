
import time
import sys

def test_imports():
    print("Testing torch import...")
    t0 = time.time()
    try:
        import torch
        print(f"Torch imported in {time.time() - t0:.2f}s")
        print(f"Torch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"Failed to import torch: {e}")

    print("\nTesting transformers import...")
    t0 = time.time()
    try:
        from transformers import pipeline
        print(f"Transformers pipeline imported in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"Failed to import transformers: {e}")

if __name__ == "__main__":
    test_imports()
