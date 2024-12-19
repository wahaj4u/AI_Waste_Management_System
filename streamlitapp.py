import os
import urllib.request
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class WasteClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the MobileNetV3 model pre-trained on ImageNet
        self.mobnet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # Feature extraction from the 'features.12' layer
        self.feature_extraction = create_feature_extractor(
            self.mobnet, return_nodes={'features.12': 'mob_feature'}
        )
        
        # Define a few custom layers for the model
        self.conv1 = nn.Conv2d(576, 300, 3)
        self.fc1 = nn.Linear(10800, 30)  # 30 output classes (waste types)
        self.dr = nn.Dropout()

    def forward(self, x):
        # Extract features from the MobileNetV3 model
        feature_layer = self.feature_extraction(x)['mob_feature']
        
        # Pass the feature through a convolutional layer and apply ReLU
        x = F.relu(self.conv1(feature_layer))
        
        # Flatten the output for feeding into the fully connected layer
        x = x.flatten(start_dim=1)
        
        # Apply dropout
        x = self.dr(x)
        
        # Pass through the final fully connected layer
        output = self.fc1(x)
        return output

def download_from_google_drive(url, destination):
    if not os.path.exists(destination):
        print("Downloading model checkpoint from Google Drive...")
        urllib.request.urlretrieve(url, destination)
        print("Download complete.")

# Google Drive direct download URL
checkpoint_url = "https://drive.google.com/uc?id=1oNibX-E2CYzcmViFNecM98NByp6g0pnE"
checkpoint_path = "train_loss_best.pt"

# Download the checkpoint if it doesn't exist locally
download_from_google_drive(checkpoint_url, checkpoint_path)

@st.cache_resource
def load_model(checkpoint_path):
    model = WasteClassificationModel()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except (EOFError, FileNotFoundError) as e:
        print(f"Error loading checkpoint: {e}")
        st.error("Checkpoint file is missing or corrupted.")
        st.stop()
    return model

model = load_model(checkpoint_path)

def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

class_names = [
    "aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans", "cardboard_boxes",
    "cardboard_packaging", "clothing", "coffee_grounds", "disposable_plastic_cutlery",
    "eggshells", "food_waste", "glass_beverage_bottles", "glass_cosmetic_containers",
    "glass_food_jars", "magazines", "newspaper", "office_paper", "paper_cups",
    "plastic_cup_lids", "plastic_detergent_bottles", "plastic_food_containers",
    "plastic_shopping_bags", "plastic_soda_bottles", "plastic_straws", "plastic_trash_bags",
    "plastic_water_bottles", "shoes", "steel_food_cans", "styrofoam_cups",
    "styrofoam_food_containers", "tea_bags"
]

st.title("Waste Classification App")
st.write("Upload an image to classify the type of waste.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Classifying...")
    predicted_class = predict(image, model)
    st.write(f"Predicted Class: **{class_names[predicted_class]}**")
