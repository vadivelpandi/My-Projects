import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("c:/Cladue ai project"))

try:
    from backend.model_manager import ModelManager
    mm = ModelManager()
    print(f"Total models tracked: {len(mm.model_names)}")
    for name in mm.model_names:
        print(f"- {name}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
