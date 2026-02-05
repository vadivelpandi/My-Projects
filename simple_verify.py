import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'backend'))

from backend.model_manager import ModelManager

# Mock start_background_loading to avoid threading/loading overhead during this check
def mock_start_background_loading(self):
    print("Skipping background loading for verification.")

ModelManager.start_background_loading = mock_start_background_loading

mm = ModelManager()
print(f"Total models defined: {len(mm.model_names)}")
for name in mm.model_names:
    print(f"- {name}")
