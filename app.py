import os
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F

# Define categories globally
categories = [
    "aerosol_cans", "aluminum_soda_cans", "cardboard_boxes", "cardboard_packaging",
    "clothing", "coffee_grounds", "disposable_plastic_cutlery", "eggshells",
    "food_waste", "glass_beverage_bottles", "glass_cosmetic_containers", "glass_food_jars",
    "magazines", "newspaper", "office_paper", "paper_cups", "plastic_cup_lids",
    "plastic_detergent_bottles", "plastic_food_containers", "plastic_shopping_bags",
    "plastic_soda_bottles", "plastic_straws", "plastic_trash_bags", "plastic_water_bottles",
    "shoes", "steel_food_cans", "styrofoam_cups", "styrofoam_food_containers", "tea_bags"
]

# Define the new model architecture
class WasteClassificationModel(nn.Module):
    def __init__(self):
        super(WasteClassificationModel, self).__init__()
        self.mobnet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.feature_extraction = create_feature_extractor(
            self.mobnet, return_nodes={'features.12': 'mob_feature'}
        )
        self.conv1 = nn.Conv2d(576, 300, 3)  # Adjusting the channels after feature extraction
        self.fc1 = nn.Linear(10800, len(categories))  # Adjusting the output to match the number of categories
        self.dr = nn.Dropout()

    def forward(self, x):
        feature_layer = self.feature_extraction(x)['mob_feature']
        x = F.relu(self.conv1(feature_layer))
        x = x.flatten(start_dim=1)
        x = self.dr(x)
        output = self.fc1(x)
        return output

# Flask app setup
app = Flask(__name__)

# Load your trained model
model_path = './train_account_best.pth'  # Replace with the actual path
model = None  # Initialize model to None before loading

try:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    print("Model loaded successfully.")
    
    # Initialize the model architecture
    model = WasteClassificationModel()  # Initialize the model class

    # Load the state dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model state dict loaded successfully.")

except Exception as e:
    print(f"Error loading model: {e}")

# Ensure model is an instance of nn.Module
if not isinstance(model, nn.Module):
    print(f"Error: model is not a PyTorch model, but of type {type(model)}.")
    model = None

# Define transformation for image preprocessing
preprocess_image = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to match model input size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image
])

# Disposal recommendation dictionary (same as before)
disposal_methods = {
    # Add all the categories and recommendations here
}

def classify_waste(segmented_image):
    if model is None:
        return None  # Model is not loaded properly
    
    # Preprocess the image
    input_tensor = preprocess_image(segmented_image).unsqueeze(0)  # Add batch dimension

    # Perform inference using the model
    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
        outputs = model(input_tensor)
    
    # Get the predicted category
    _, predicted_index = torch.max(outputs, 1)
    category = categories[predicted_index.item()]  # Assuming categories is a list of class names
    
    return category

# Mock functions for segmentation
def segment_waste(image):
    segmented_image = np.array(image)  # Example segmentation (replace with actual logic)
    
    if segmented_image.shape[-1] == 4:  # Convert to RGB if image has 4 channels
        segmented_image = segmented_image[:, :, :3]
    
    return Image.fromarray(segmented_image)  # Convert back to PIL Image

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = Image.open(image_file)

    # Segment the waste
    segmented_image = segment_waste(image)

    # Classify the waste
    category = classify_waste(segmented_image)

    if category is None:
        return jsonify({'error': 'Model not loaded or invalid image.'}), 500

    # Get disposal recommendation
    disposal_info = disposal_methods.get(category, {"recommendation": "No information available.", "type": "unknown"})

    return jsonify({
        "category": category,
        "disposal_recommendation": disposal_info["recommendation"],
        "type": disposal_info["type"]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
