import os
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models


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

# Define the model architecture
class WasteClassificationModel(nn.Module):
    def __init__(self):
        super(WasteClassificationModel, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256),  # Adjust based on MobileNetV2 output channel (1280)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(categories))  # Set the output layer to match number of categories
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x

# Flask app setup
app = Flask(__name__)

# Load your trained model
model_path = './train_account_best.pth'  # Replace with the actual path
model = None  # Initialize model to None before loading

try:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    print("Model loaded successfully.")
    print(f"Model dictionary keys: {checkpoint.keys()}")
    
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
    transforms.Resize((224, 224)),  # Resize image to match model input size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])







# Disposal recommendation dictionary
disposal_methods = {
    "aerosol_cans": {
        "recommendation": "Make sure the can is empty before disposal. Check with your local recycling program for acceptance. If not recyclable, dispose of as hazardous waste.",
        "type": "Hazardous Waste",
        "recyclable": "No",
        "compostable": "No"
    },
    "aluminum_food_cans": {
        "recommendation": "Rinse the can thoroughly to remove any food residue. Place it in your recycling bin. Crushing the can saves space but is optional.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "aluminum_soda_cans": {
        "recommendation": "Rinse to remove sticky residue. Place the can in your recycling bin. Avoid crushing if your recycling program requires intact cans.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "cardboard_boxes": {
        "recommendation": "Flatten the box to save space before recycling. Remove any non-cardboard elements like tape or labels. Place in the recycling bin for paper/cardboard.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "cardboard_packaging": {
        "recommendation": "Ensure all packaging is flattened for easy recycling. Remove non-cardboard parts such as plastic film or foam. Recycle with other cardboard materials.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "clothing": {
        "recommendation": "If still wearable, consider donating to local charities or thrift stores. For damaged clothing, recycle as fabric or take to textile recycling bins. Avoid placing in general waste.",
        "type": "Recyclable/Textile Recycling",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "coffee_grounds": {
        "recommendation": "Coffee grounds are rich in nutrients and can be composted. Add them to your compost bin or garden soil. If composting is not an option, dispose of them in organic waste bins.",
        "type": "Compostable",
        "recyclable": "No",
        "compostable": "Yes"
    },
    "disposable_plastic_cutlery": {
        "recommendation": "Most disposable cutlery is not recyclable. Place it in the general waste bin. Consider switching to reusable or compostable alternatives in the future.",
        "type": "General Waste",
        "recyclable": "No",
        "compostable": "No"
    },
    "eggshells": {
        "recommendation": "Eggshells can be composted and are great for enriching soil. Add them to your compost bin after rinsing. Alternatively, place in organic waste bins.",
        "type": "Compostable",
        "recyclable": "No",
        "compostable": "Yes"
    },
    "food_waste": {
        "recommendation": "Separate food waste from packaging before disposal. Compost if possible to reduce landfill impact. Use organic waste bins where available.",
        "type": "Compostable",
        "recyclable": "No",
        "compostable": "Yes"
    },
    "glass_beverage_bottles": {
        "recommendation": "Rinse thoroughly to remove any liquid. Place in the glass recycling bin. Remove caps or lids if not made of glass.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "glass_cosmetic_containers": {
        "recommendation": "Clean the container to ensure it's residue-free. Recycle if your local program accepts glass containers. Broken glass should be wrapped in paper or cardboard and placed in general waste.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "glass_food_jars": {
        "recommendation": "Rinse the jar to remove food residue. Recycle in glass bins. Lids made of metal can often be recycled separately.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "magazines": {
        "recommendation": "Remove plastic covers or non-paper elements before recycling. Place in your paper recycling bin. Avoid recycling if excessively wet or damaged.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "newspaper": {
        "recommendation": "Keep newspapers dry and free of contaminants like food stains. Recycle them in designated paper bins. Bundle them for easier handling if required.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "office_paper": {
        "recommendation": "Shred confidential documents if necessary before recycling. Avoid including paper with heavy lamination or plastic content. Recycle in paper bins.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "paper_cups": {
        "recommendation": "Check for a recycling symbol to confirm if recyclable. Most paper cups with plastic lining are not recyclable and go into general waste. Consider switching to reusable cups.",
        "type": "General Waste",
        "recyclable": "No",
        "compostable": "No"
    },
    "plastic_cup_lids": {
        "recommendation": "If marked recyclable, clean and place them in the appropriate bin. Otherwise, dispose of in general waste. Avoid using single-use lids when possible.",
        "type": "Recyclable/General Waste",
        "recyclable": "Yes/No",
        "compostable": "No"
    },
    "plastic_detergent_bottles": {
        "recommendation": "Rinse out any remaining detergent to avoid contamination. Check the recycling symbol and place in plastics recycling. Keep the lid on if acceptable.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "plastic_food_containers": {
        "recommendation": "Ensure the container is clean and free of food residue. Recycle if marked as recyclable. Otherwise, dispose of in general waste.",
        "type": "Recyclable/General Waste",
        "recyclable": "Yes/No",
        "compostable": "No"
    },
    "plastic_shopping_bags": {
        "recommendation": "Reuse them for storage or garbage liners. If recycling facilities for plastic bags are available, drop them off. Avoid throwing in general recycling bins.",
        "type": "Recyclable/General Waste",
        "recyclable": "Yes/No",
        "compostable": "No"
    },
    "plastic_soda_bottles": {
        "recommendation": "Empty and rinse the bottle before recycling. Leave the cap on if your recycling program accepts it. Crush the bottle to save space if desired.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "plastic_straws": {
        "recommendation": "Plastic straws are not recyclable in most programs. Dispose of them in general waste. Consider using reusable or biodegradable straws.",
        "type": "General Waste",
        "recyclable": "No",
        "compostable": "No"
    },
    "plastic_trash_bags": {
        "recommendation": "Trash bags themselves are not recyclable. Dispose of them in general waste along with their contents. Look for biodegradable options when purchasing new ones.",
        "type": "General Waste",
        "recyclable": "No",
        "compostable": "No"
    },
    "plastic_water_bottles": {
        "recommendation": "Rinse the bottle to ensure cleanliness. Recycle the bottle along with the cap if accepted. Try to use reusable bottles to reduce plastic waste.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "shoes": {
        "recommendation": "Donate shoes that are still wearable to charities or thrift stores. For damaged or unusable shoes, check for textile recycling bins. Avoid discarding in general waste.",
        "type": "Recyclable/Textile Recycling",
        "recyclable": "Yes",
        "compostable":"No"
    }
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
    
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

    #app.run(debug=True)
