import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.feature_extraction import create_feature_extractor

# Define the WasteClassificationModel class
class WasteClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the MobileNetV3 with pretrained weights
        self.mobnet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.feature_extraction = create_feature_extractor(
            self.mobnet, return_nodes={'features.12': 'mob_feature'}
        )
        self.conv1 = nn.Conv2d(576, 300, 3)
        self.fc1 = nn.Linear(10800, 30)
        self.dr = nn.Dropout()

    def forward(self, x):
        feature_layer = self.feature_extraction(x)['mob_feature']
        x = F.relu(self.conv1(feature_layer))
        x = x.flatten(start_dim=1)
        x = self.dr(x)
        output = self.fc1(x)
        return output

# Load the model and checkpoint
@st.cache_resource
def load_model(checkpoint_path):
    model = WasteClassificationModel()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Path to the saved model file
model = load_model("train_loss_best.pt")

# Define the prediction function
def predict(image, model):
    transform = transforms.Compose([  # Correct usage of Compose
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Define the class names
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

# Streamlit App UI
st.title("Waste Classification App")
st.write("Upload an image to classify the type of waste.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Prediction
    st.write("Classifying...")
    predicted_class = predict(image, model)
    st.write(f"Predicted Class: **{class_names[predicted_class]}**")
