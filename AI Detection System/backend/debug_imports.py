
print("Start Debug Imports")
print("Importing os...")
import os
print("Importing time...")
import time
print("Importing cv2...")
import cv2
print("Importing numpy...")
import numpy as np
print("Importing PIL...")
from PIL import Image
print("Importing torch...")
import torch
print("Importing transformers...")
try:
    from transformers import pipeline
    print("Transformers imported successfully.")
except Exception as e:
    print(f"Transformers import failed: {e}")

print("Importing ModelManager...")
try:
    from model_manager import ModelManager
    print("ModelManager imported.")
except Exception as e:
    print(f"ModelManager import failed: {e}")

print("Importing ForensicEngine...")
try:
    from forensic_engine import ForensicEngine
    print("ForensicEngine imported.")
except Exception as e:
    print(f"ForensicEngine import failed: {e}")

print("Importing MetadataEngine...")
try:
    from metadata_engine import MetadataEngine
    print("MetadataEngine imported.")
except Exception as e:
    print(f"MetadataEngine import failed: {e}")

print("DONE.")
