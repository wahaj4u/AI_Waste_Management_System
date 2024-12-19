import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import requests
from sam import SamPredictor

# Load SAM model
sam_model_path = 'path_to_sam_vit_b.pth'  # Path to the SAM model .pth file
sam_model = SamModel.from_pretrained(sam_model_path)
sam_model.to('cuda' if torch.cuda.is_available() else 'cpu')
predictor = SamPredictor(sam_model)

# Load MobileNetV2
mobilenet_v2_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
mobilenet_v2_model.eval()

# Disposal recommendation dictionary
disposal_methods = {
    "aerosol_cans": "Make sure the can is empty before disposal. Check with your local recycling program for acceptance...",
    # Add all disposal methods as given in your dictionary...
}

def classify_image(image):
    # Transform image for MobileNetV2
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Transform and prepare image
    input_tensor = preprocess(image).unsqueeze(0)
    input_tensor = input_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Predict using MobileNetV2
    with torch.no_grad():
        outputs = mobilenet_v2_model(input_tensor)
    
    # Get predicted class
    _, predicted_class_idx = outputs.max(1)
    return predicted_class_idx.item()

def main():
    st.title("Waste Sorting Application")
    
    # Capture image
    if st.button('Capture Image'):
        image_file = st.camera_input("Capture a Waste Image")
        if image_file:
            image = Image.open(image_file)

            # Segment using SAM2
            predictor.set_image(image)
            masks, scores, logits = predictor.predict('cpu')

            # Display segmented image
            st.image(masks, caption="Segmented Image")

            # Classify using MobileNetV2
            class_idx = classify_image(masks)
            category = mobilenet_v2_model.classes[class_idx]  # Map index to category
            
            # Provide disposal recommendation
            recommendation = disposal_methods.get(category, "No disposal information available.")
            st.write(f"Disposal Recommendation: {recommendation}")

if __name__ == '__main__':
    main()
