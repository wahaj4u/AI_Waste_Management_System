import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image

# Load the trained model from your GitHub repository
model_url = "https://github.com/wahaj4u/AI_Waste_Management_System/blob/main/train_account_best.pth"
model_path = "train_account_best.pth"
torch.hub.download_url_to_file(model_url, model_path)

# Attempt to load the model and catch any exceptions
try:
    model = torch.load(model_path)
    model.eval()
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

# Image transformations
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Title of the Streamlit App
st.title("Waste Classification App")
st.write("Capture an image using your camera to classify the waste and get disposal instructions.")

# Disposal recommendation dictionary
disposal_recommendations = {
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
    # Add the rest of the categories...
}

# Camera input section
captured_image = st.camera_input("Capture an image using your camera")

if captured_image is not None:
    try:
        # Display the captured image
        image = Image.open(captured_image)
        st.image(image, caption="Captured Image", use_column_width=True)

        # Preprocess the image
        input_image = preprocess(image)
        input_image = input_image.unsqueeze(0)  # Add batch dimension

        # Run the model on the image
        with torch.no_grad():
            outputs = model(input_image)
        
        # Get the category with the highest score
        _, predicted_idx = torch.max(outputs, 1)
        category = model.classes[predicted_idx.item()].lower()
        
        # Fetch disposal information
        waste_info = disposal_recommendations.get(category, {
            "recommendation": "No specific instructions available.",
            "type": "Unknown",
            "recyclable": "Unknown",
            "compostable": "Unknown"
        })
        
        # Display the results
        st.write("### Classification Result:")
        st.write(f"**Category**: {category}")
        st.write(f"**Type**: {waste_info['type']}")
        st.write(f"**Recyclable**: {waste_info['recyclable']}")
        st.write(f"**Compostable**: {waste_info['compostable']}")
        st.write(f"**Disposal Recommendation**: {waste_info['recommendation']}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
