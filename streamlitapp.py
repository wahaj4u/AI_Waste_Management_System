import torch
from segment_anything import sam_model_registry

checkpoint_path = 'sam_vit_b.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry['vit_b'](checkpoint=checkpoint_path)
sam.to(device)
print("Model loaded successfully!")
