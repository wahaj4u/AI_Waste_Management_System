import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Define disposal recommendations
disposal_methods = {
    "aerosol_cans": "Ensure empty before disposal. Check local recycling or dispose as hazardous waste.",
    "aluminum_food_cans": "Rinse, remove food residue, and recycle.",
    "cardboard_boxes": "Flatten and recycle. Remove tape/labels.",
    # Add other classes as needed
}

# Initialize SAM model
@st.cache_resource
def load_sam_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry['vit_b'](checkpoint='/path/to/sam_vit_b.pth')
    sam.to(device)
    return SamAutomaticMaskGenerator(sam)

# Load trained classification model
@st.cache_resource
def load_classification_model():
    num_classes = len(disposal_methods)
    model = torch.load('./train_account_best.pth', map_location=torch.device('cpu'))
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

# Main Streamlit App
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
        mask_generator = load_sam_model()

        image_np = np.array(image)
        masks = mask_generator.generate(image_np)

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

if __name__ == "__main__":
    main()
