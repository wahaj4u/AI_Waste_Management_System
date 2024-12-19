import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the pre-trained MobileNetV2 model for waste classification
model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

# Dictionary for disposal recommendation based on category
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
    # Add more items as needed
}

# Title of the Streamlit App
st.title("Waste Classification App")
st.write("Capture an image using your camera to classify the waste and get disposal instructions.")

# Camera input section
captured_image = st.camera_input("Capture an image using your camera")

if captured_image is not None:
    try:
        # Display the captured image
        image = Image.open(captured_image)
        
        # Resize the image for MobileNetV2 input
        image_resized = image.resize((224, 224))  # Resize to 224x224 for standard image sizes used in deep learning models
        
        st.image(image_resized, caption="Resized Captured Image", use_column_width=True)

        # Preprocess the image for MobileNetV2
        img_array = np.array(image_resized)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_array, axis=0))

        # Predict using the model
        preds = model.predict(img_array)
        predictions = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]

        # Get the predicted category
        category = predictions[0][1].lower()  # Convert to lowercase for consistency

        # Fetch disposal information
        waste_info = disposal_methods.get(category, {
            "recommendation": "No specific instructions available.",
            "type": "Unknown",
            "recyclable": "Unknown",
            "compostable": "Unknown"
        })

        # Display the results
        st.write("### Classification Result:")
        st.write(f"**Category**: {predictions[0][1]}")
        st.write(f"**Type**: {waste_info['type']}")
        st.write(f"**Recyclable**: {waste_info['recyclable']}")
        st.write(f"**Compostable**: {waste_info['compostable']}")
        st.write(f"**Disposal Recommendation**: {waste_info['recommendation']}")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
