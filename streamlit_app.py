# Import necessary libraries
import streamlit as st
from PIL import Image
import torch
from model import WasteClassificationModel  # Import your model class

# Title of the Streamlit App
st.title("Waste Classification App")
st.write("Capture an image using your camera to classify the waste and get disposal instructions.")

# Function to load the model
def load_model():
    model = WasteClassificationModel()
    checkpoint = torch.load("train_account_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model

# Disposal recommendation dictionary
disposal_methods = {
    "aerosol_cans": {
        "recommendation": "Make sure the can is empty before disposal. Check with your local recycling program for acceptance. If not recyclable, dispose of as hazardous waste.",
        "type": "Hazardous",
        "recyclable": "Partially Recyclable",
        "compostable": "No"
    },
    "aluminum_food_cans": {
        "recommendation": "Rinse the can thoroughly to remove any food residue. Place it in your recycling bin. Crushing the can saves space but is optional.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "coffee_grounds": {
        "recommendation": "Coffee grounds are rich in nutrients and can be composted. Add them to your compost bin or garden soil. If composting is not an option, dispose of them in organic waste bins.",
        "type": "Compostable",
        "recyclable": "No",
        "compostable": "Yes"
    },
    "plastic_soda_bottles": {
        "recommendation": "Empty and rinse the bottle before recycling. Leave the cap on if your recycling program accepts it. Crush the bottle to save space if desired.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    # Add additional items as needed
}

# Camera input section
captured_image = st.camera_input("Capture an image using your camera")

if captured_image is not None:
    try:
        # Display the captured image
        image = Image.open(captured_image)
        st.image(image, caption="Captured Image", use_column_width=True)

        # Convert the captured image to tensor for the model
        image_tensor = data_transforms["test"](image).unsqueeze(0)  # Batch size 1

        # Load the model
        model = load_model()

        # Classify the image
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            category = class_names[predicted.item()].lower()
            
            # Fetch disposal information
            waste_info = disposal_methods.get(category, {
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
        st.error(f"An unexpected error occurred: {str(e)}")
