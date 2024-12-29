import streamlit as st
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os

# Load SAM model
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
        st.error(f"Error loading SAM model: {e}")
        raise

# Streamlit App
def main():
    st.title("SAM Model Streamlit App")

    # Image Upload
    st.subheader("Upload an Image for Segmentation")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the image to NumPy array for processing
        image_np = np.array(image)

        # Step 1: Load SAM Model and generate segmentation masks
        st.subheader("Generating Segmentation Masks")
        try:
            mask_generator = load_sam_model()
            masks = mask_generator.generate(image_np)

            if masks:
                mask = masks[0]['segmentation']  # Use the first mask
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                st.image(mask_image, caption="Segmented Mask", use_column_width=True)
            else:
                st.error("No segmentation mask was generated.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
if __name__ == "__main__":
    main()
