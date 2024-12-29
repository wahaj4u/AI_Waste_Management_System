import torch
from segment_anything import sam_model_registry

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model checkpoint with weights only
checkpoint_path = 'sam_vit_b.pth'
sam = sam_model_registry['vit_b'](checkpoint=checkpoint_path, weights_only=True)
sam.to(device)

print("Model loaded successfully!")
