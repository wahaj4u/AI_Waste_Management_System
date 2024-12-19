import streamlit as st
import requests
from PIL import Image

# Title of the Streamlit App
st.title("Waste Classification App")
st.write("Capture an image using your camera to classify the waste and get disposal instructions.")

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
    "paper_towels": {
        "recommendation": "Tear into smaller pieces and place in the compost bin. Do not dispose of in recycling.",
        "type": "Compostable",
        "recyclable": "No",
        "compostable": "Yes"
    },
    "cardboard_boxes": {
        "recommendation": "Flatten the boxes and place them in your recycling bin. Remove any tape and packing materials.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "glass_jars": {
        "recommendation": "Rinse out jars and place them in your recycling bin. If broken, dispose of in your waste bin as glass can be hazardous.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "styrofoam_trays": {
        "recommendation": "Unfortunately, Styrofoam is not recyclable. Dispose of it in your waste bin.",
        "type": "Non-Recyclable",
        "recyclable": "No",
        "compostable": "No"
    },
    "electronics": {
        "recommendation": "Take electronics to a local e-waste recycling facility. Do not dispose of them in regular trash bins.",
        "type": "Hazardous",
        "recyclable": "No",
        "compostable": "No"
    },
    "textiles": {
        "recommendation": "Donate or recycle old clothes. Do not dispose of them in regular waste bins.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    }
    # Add additional items as needed
}

# Camera input section
captured_image = st.camera_input("Capture an image using your camera")

if captured_image is not None:
    try:
        # Display the captured image
        image = Image.open(captured_image)
        
        # Resize the image for quicker processing
        image_resized = image.resize((224, 224))  # Resize to 224x224 for standard image sizes used in deep learning models
        
        st.image(image_resized, caption="Resized Captured Image", use_column_width=True)

        # Convert the resized image to bytes for the POST request
        img_byte_arr = image_resized.tobytes()
        files = {'image': img_byte_arr}

        # Replace with the active ngrok or backend URL
        backend_url = "https://9c9c-34-138-89-45.ngrok-free.app//process"

        # Send the image to the Flask backend
        response = requests.post(backend_url, files=files)

        if response.status_code == 200:
            result = response.json()
            category = result.get('category', 'unknown').lower()
            
            # Fetch disposal information
            waste_info = disposal_methods.get(category, {
                "recommendation": "No specific instructions available.",
                "type": "Unknown",
                "recyclable": "Unknown",
                "compostable": "Unknown"
            })
            
            # Display the results
            st.write("### Classification Result:")
            st.write(f"**Category**: {result.get('category', 'N/A')}")
            st.write(f"**Type**: {waste_info['type']}")
            st.write(f"**Recyclable**: {waste_info['recyclable']}")
            st.write(f"**Compostable**: {waste_info['compostable']}")
            st.write(f"**Disposal Recommendation**: {waste_info['recommendation']}")
        else:
            st.error(f"Error: Unable to classify the image. Status code: {response.status_code}")
            st.error(f"Response: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the backend. Please ensure the backend is running and the URL is correct.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
