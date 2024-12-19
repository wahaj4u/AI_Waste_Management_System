import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# Load the trained model
@st.cache_resource
def load_model():
    model = WasteClassificationModel()  # Replace with your model class
    checkpoint = torch.load("train_account_best.pth", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# Preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(image).unsqueeze(0)

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

# Initialize Streamlit App
st.title("Waste Classification App")
st.write("Capture an image using your camera to classify the waste and get disposal instructions.")

# Camera input
captured_image = st.camera_input("Capture an image using your camera")

if captured_image is not None:
    model = load_model()  # Load the model
    class_names = ["aerosol_cans", "aluminum_food_cans", "coffee_grounds", "plastic_soda_bottles"]  # Update based on your classes

    # Display the captured image
    image = Image.open(captured_image)
    st.image(image, caption="Captured Image", use_column_width=True)

    # Preprocess the image
    st.write("Classifying the waste...")
    input_tensor = preprocess_image(image)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, prediction = torch.max(outputs, 1)
        category = class_names[prediction.item()]

    # Fetch disposal information
    waste_info = disposal_methods.get(category, {
        "recommendation": "No specific instructions available.",
        "type": "Unknown",
        "recyclable": "Unknown",
        "compostable": "Unknown"
    })

    # Display classification results
    st.write("### Classification Result:")
    st.write(f"**Category**: {category}")
    st.write(f"**Type**: {waste_info['type']}")
    st.write(f"**Recyclable**: {waste_info['recyclable']}")
    st.write(f"**Compostable**: {waste_info['compostable']}")
    st.write(f"**Disposal Recommendation**: {waste_info['recommendation']}")

    # Display additional metrics (optional)
    if st.button("Show Test Metrics"):
        # Example metrics display
        y_preds = [0, 1, 2, 3]  # Replace with real predictions
        y_true = [0, 1, 2, 3]   # Replace with real ground truths
        st.write("\n-------Classification Report-------\n")
        st.text(classification_report(y_true, y_preds, target_names=class_names))
        cm = np.array([[5, 0, 0, 0], [0, 6, 0, 0], [0, 0, 7, 0], [0, 0, 0, 8]])  # Example confusion matrix
        fig, ax = plt.subplots(figsize=(6, 6))
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        cm_disp.plot(ax=ax)
        st.pyplot(fig)
