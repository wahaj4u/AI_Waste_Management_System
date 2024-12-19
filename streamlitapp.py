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
    # Add additional items as needed
}

# Camera input section
captured_image = st.camera_input("Capture an image using your camera")

if captured_image is not None:
    try:
        # Display the captured image
        image = Image.open(captured_image)
        st.image(image, caption="Captured Image", use_column_width=True)

        # Convert the captured image to bytes for the POST request
        st.write("Classifying the waste...")
        files = {"image": captured_image.getvalue()}
        
        # Replace with the active ngrok or backend URL
        backend_url = "https://5021-34-138-89-45.ngrok-free.app/process"

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
