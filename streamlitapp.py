import streamlit as st
from PIL import Image
import numpy as np
import torch
import os
from torchvision import transforms
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision.transforms import ToTensor

# Define disposal methods dictionary
disposal_methods = {
    'plastic': "Recycle",
    'paper': "Recycle",
    'metal': "Recycle",
    'glass': "Recycle",
    'food': "Compost",
    'clothing': "Donate",
    'electronic': "Special Disposal",
    'hazardous': "Special Disposal",
    'general': "Landfill"
}

# Preprocess image for classification model
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image)

# Load classification model (dummy model for example purposes)
def load_classification_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.eval()
    return model

# Load SAM model and create a mask generator
def load_sam_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'sam_vit_b.pth')

    try:
        # Load the SAM model
        sam = sam_model_registry['vit_b'](checkpoint=checkpoint_path)
        sam.to(device)

        # Create a mask generator using the model
        mask_generator = SamAutomaticMaskGenerator(sam)
        return mask_generator

    except Exception as e:
        print(f"Error loading the SAM model: {e}")
        raise

# Main function to handle app flow
def main():
    st.title("WasteSort AI: Waste Sorting and Disposal Assistant")

    # Step 1: Capture image
    st.subheader("Step 1: Capture an Image")
    uploaded_image = st.camera_input("Take a picture")

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Step 2: Segment the image
        st.subheader("Step 2: Segmenting the Image")
        
        try:
            mask_generator = load_sam_model()  # Load SAM model and generator

            image_np = np.array(image)  # Convert image to NumPy array
            masks = mask_generator.generate(image_np)  # Generate segmentation masks

            if masks:
                mask = masks[0]['segmentation']  # Use the first mask
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                st.image(mask_image, caption="Segmented Mask", use_column_width=True)

                # Step 3: Classify the segmented object
                st.subheader("Step 3: Classifying the Object")

                # Convert the mask to 1 channel
                mask_tensor = ToTensor()(mask_image).unsqueeze(0)

                # Preprocess original image
                image_tensor = preprocess_image(image).unsqueeze(0)

                # Concatenate the mask with the image
                input_tensor = torch.cat([image_tensor, mask_tensor], dim=1)

                model = load_classification_model()
                with torch.no_grad():
                    outputs = model(input_tensor)
                    predicted_class_idx = torch.argmax(outputs, dim=1).item()
                    predicted_class = list(disposal_methods.keys())[predicted_class_idx]

                # Step 4: Display disposal recommendation
                st.subheader("Step 4: Disposal Recommendation")
                recommendation = disposal_methods.get(predicted_class, "No recommendation available.")
                st.write(f"**Classified as**: {predicted_class}")
                st.write(f"**Disposal Recommendation**: {recommendation}")
            else:
                st.error("No segmentation mask could be generated.")
        
        except Exception as e:
            st.error(f"An error occurred during segmentation: {e}")
        
if __name__ == "__main__":
    main()
