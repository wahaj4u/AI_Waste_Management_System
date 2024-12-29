import streamlit as st
import torch
from PIL import Image
import cv2
import os
import requests
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Define disposal recommendations
disposal_methods = {
    "aerosol_cans": "Make sure the can is empty before disposal. Check with your local recycling program for acceptance. If not recyclable, dispose of as hazardous waste.",
    "aluminum_food_cans": "Rinse the can thoroughly to remove any food residue. Place it in your recycling bin. Crushing the can saves space but is optional.",
    "aluminum_soda_cans": "Rinse to remove sticky residue. Place the can in your recycling bin. Avoid crushing if your recycling program requires intact cans.",
    "cardboard_boxes": "Flatten the box to save space before recycling. Remove any non-cardboard elements like tape or labels. Place in the recycling bin for paper/cardboard.",
    "cardboard_packaging": "Ensure all packaging is flattened for easy recycling. Remove non-cardboard parts such as plastic film or foam. Recycle with other cardboard materials.",
    "clothing": "If still wearable, consider donating to local charities or thrift stores. For damaged clothing, recycle as fabric or take to textile recycling bins. Avoid placing in general waste.",
    "coffee_grounds": "Coffee grounds are rich in nutrients and can be composted. Add them to your compost bin or garden soil. If composting is not an option, dispose of them in organic waste bins.",
    "disposable_plastic_cutlery": "Most disposable cutlery is not recyclable. Place it in the general waste bin. Consider switching to reusable or compostable alternatives in the future.",
    "eggshells": "Eggshells can be composted and are great for enriching soil. Add them to your compost bin after rinsing. Alternatively, place in organic waste bins.",
    "food_waste": "Separate food waste from packaging before disposal. Compost if possible to reduce landfill impact. Use organic waste bins where available.",
    "glass_beverage_bottles": "Rinse thoroughly to remove any liquid. Place in the glass recycling bin. Remove caps or lids if not made of glass.",
    "glass_cosmetic_containers": "Clean the container to ensure it's residue-free. Recycle if your local program accepts glass containers. Broken glass should be wrapped in paper or cardboard and placed in general waste.",
    "glass_food_jars": "Rinse the jar to remove food residue. Recycle in glass bins. Lids made of metal can often be recycled separately.",
    "magazines": "Remove plastic covers or non-paper elements before recycling. Place in your paper recycling bin. Avoid recycling if excessively wet or damaged.",
    "newspaper": "Keep newspapers dry and free of contaminants like food stains. Recycle them in designated paper bins. Bundle them for easier handling if required.",
    "office_paper": "Shred confidential documents if necessary before recycling. Avoid including paper with heavy lamination or plastic content. Recycle in paper bins.",
    "paper_cups": "Check for a recycling symbol to confirm if recyclable. Most paper cups with plastic lining are not recyclable and go into general waste. Consider switching to reusable cups.",
    "plastic_cup_lids": "If marked recyclable, clean and place them in the appropriate bin. Otherwise, dispose of in general waste. Avoid using single-use lids when possible.",
    "plastic_detergent_bottles": "Rinse out any remaining detergent to avoid contamination. Check the recycling symbol and place in plastics recycling. Keep the lid on if acceptable.",
    "plastic_food_containers": "Ensure the container is clean and free of food residue. Recycle if marked as recyclable. Otherwise, dispose of in general waste.",
    "plastic_shopping_bags": "Reuse them for storage or garbage liners. If recycling facilities for plastic bags are available, drop them off. Avoid throwing in general recycling bins.",
    "plastic_soda_bottles": "Empty and rinse the bottle before recycling. Leave the cap on if your recycling program accepts it. Crush the bottle to save space if desired.",
    "plastic_straws": "Plastic straws are not recyclable in most programs. Dispose of them in general waste. Consider using reusable or biodegradable straws.",
    "plastic_trash_bags": "Trash bags themselves are not recyclable. Dispose of them in general waste along with their contents. Look for biodegradable options when purchasing new ones.",
    "plastic_water_bottles": "Rinse the bottle to ensure cleanliness. Recycle the bottle along with the cap if accepted. Try to use reusable bottles to reduce plastic waste.",
    "shoes": "Donate shoes that are still wearable to charities or thrift stores. For damaged or unusable shoes, check for textile recycling bins. Avoid discarding in general waste.",
    "steel_food_cans": "Clean the can by removing all food residue. Place it in your recycling bin. Check for local recycling guidelines if needed.",
    "styrofoam_cups": "Styrofoam is not recyclable in most areas. Dispose of it in general waste. Avoid using Styrofoam products whenever possible.",
    "styrofoam_food_containers": "Clean the container before disposal if required. Place it in general waste as Styrofoam is typically non-recyclable. Consider switching to sustainable alternatives.",
    "tea_bags": "Compost biodegradable tea bags as they are rich in organic matter. Check if your tea bags have plastic components and dispose of those in general waste."
}

# Function to download the SAM model if it doesn't exist locally
def download_sam_model():
    model_url = "https://github.com/wahaj4u/AI_Waste_Management_System/releases/download/sammodel/sam_vit_b.pth"
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'sam_vit_b.pth')

    if not os.path.exists(checkpoint_path):
        st.info("Downloading SAM model. This may take a moment...")
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with open(checkpoint_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success("SAM model downloaded successfully.")
        else:
            st.error(f"Failed to download SAM model. Status code: {response.status_code}")
            return None
    return checkpoint_path

# Initialize SAM model
@st.cache_resource
def load_sam_model():
    checkpoint_path = download_sam_model()
    if checkpoint_path is None:
        st.error("Could not load SAM model.")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry['vit_b'](checkpoint=checkpoint_path)
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

        if mask_generator is None:
            st.error("SAM model not loaded. Please check the configuration.")
            return

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
