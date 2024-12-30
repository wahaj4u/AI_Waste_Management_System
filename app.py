from flask import Flask, request, jsonify
import torch
from PIL import Image
import numpy as np
import os
import requests
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

app = Flask(__name__)

# Define disposal recommendations
disposal_methods = {
    "aerosol_cans": "Make sure the can is empty before disposal. Check with your local recycling program for acceptance. If not recyclable, dispose of as hazardous waste.",
    "aluminum_food_cans": "Rinse the can thoroughly to remove any food residue. Place it in your recycling bin. Crushing the can saves space but is optional.",
    "aluminum_soda_cans": "Rinse to remove sticky residue. Place the can in your recycling bin. Avoid crushing if your recycling program requires intact cans.",
    "cardboard_boxes": "Flatten the box to save space before recycling. Remove any non-cardboard elements like tape or labels. Place in the recycling bin for paper/cardboard.",
    # Add the rest of the disposal methods here...
}

# Function to download the SAM model if it doesn't exist locally
def download_sam_model():
    model_url = "https://github.com/wahaj4u/AI_Waste_Management_System/releases/download/sammodel/model.pth"
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'sam_vit_b.pth')

    if not os.path.exists(checkpoint_path):
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with open(checkpoint_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            raise RuntimeError(f"Failed to download SAM model. Status code: {response.status_code}")
    return checkpoint_path

# Load the SAM model
def load_sam_model():
    checkpoint_path = download_sam_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry['vit_b'](checkpoint=checkpoint_path)
    sam.to(device)
    return SamAutomaticMaskGenerator(sam)

# Load trained classification model
def load_classification_model():
    model_path = './train_account_best.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classification model not found at {model_path}.")
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

# Preprocessing transformations
def preprocess_image(image):
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)

# Initialize models
sam_model = load_sam_model()
classification_model = load_classification_model()

@app.route('/classify', methods=['POST'])
def classify_waste():
    try:
        # Check if an image file is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided."}), 400
        
        image_file = request.files['image']
        image = Image.open(image_file)

        # Generate segmentation mask
        image_np = np.array(image)
        masks = sam_model.generate(image_np)

        if not masks:
            return jsonify({"error": "No segmentation mask could be generated."}), 500

        mask = masks[0]['segmentation']
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))

        # Preprocess image and mask
        mask_tensor = ToTensor()(mask_image).unsqueeze(0)
        image_tensor = preprocess_image(image).unsqueeze(0)
        input_tensor = torch.cat([image_tensor, mask_tensor], dim=1)

        # Classify the object
        with torch.no_grad():
            outputs = classification_model(input_tensor)
            predicted_class_idx = torch.argmax(outputs, dim=1).item()
            predicted_class = list(disposal_methods.keys())[predicted_class_idx]

        # Return the result
        recommendation = disposal_methods.get(predicted_class, "No recommendation available.")
        return jsonify({
            "class": predicted_class,
            "recommendation": recommendation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
