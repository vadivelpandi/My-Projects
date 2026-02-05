import torch
import traceback
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    print("Attempting to load via torch.hub...")
    # Loading the 'blur_jpg' model which is one of the robust variants, or 'progan' (baseline)
    model = torch.hub.load('peterwang512/CNNDetection', 'wang2020', 'progan', pretrained=True, trust_repo=True)
    model.eval()
    print("Successfully loaded via torch.hub!")
    
    # Test valid transforms
    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("Transforms defined.")
    
except Exception as e:
    print(f"Failed loading via torch.hub: {e}")
    traceback.print_exc()
