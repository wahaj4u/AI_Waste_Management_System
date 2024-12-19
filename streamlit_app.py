import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Title of the Streamlit App
st.title("Plastic Waste Classification App")
st.write("Capture an image using your camera to classify plastic soda bottles and get disposal instructions.")

# Disposal recommendation dictionary
disposal_methods = {
    "plastic_soda_bottles": {
        "recommendation": "Empty and rinse the bottle before recycling. Leave the cap on if your recycling program accepts it. Crush the bottle to save space if desired.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
}

# Camera input section
captured_image = st.camera_input("Capture an image using your camera")

if captured_image is not None:
    try:
        # Display the captured image
        image = Image.open(captured_image)
        st.image(image, caption="Captured Image", use_column_width=True)

        # Convert the captured image to bytes for the POST request
        st.write("Processing the image for segmentation and classification...")
        files = {"image": captured_image.getvalue()}

        # Replace with the active ngrok or backend URL
        backend_url = "https://5021-34-138-89-45.ngrok-free.app/process"

        # Send the image to the Flask backend
        response = requests.post(backend_url, files=files)

        if response.status_code == 200:
            result = response.json()

            # Fetch segmented image
            segmented_image_data = result.get('segmented_image')
            category = result.get('category', '').strip().lower()

            if segmented_image_data:
                # Convert segmented image bytes to displayable format
                segmented_image = Image.open(BytesIO(segmented_image_data))
                st.image(segmented_image, caption="Segmented Image", use_column_width=True)

            if category == "plastic_soda_bottles":
                waste_info = disposal_methods.get(category, {
                    "recommendation": "No specific instructions available.",
                    "type": "Unknown",
                    "recyclable": "Unknown",
                    "compostable": "Unknown"
                })

                # Display the classification results
                st.write("### Classification Result:")
                st.write(f"**Category**: {result.get('category', 'N/A')}")
                st.write(f"**Type**: {waste_info['type']}")
                st.write(f"**Recyclable**: {waste_info['recyclable']}")
                st.write(f"**Compostable**: {waste_info['compostable']}")
                st.write(f"**Disposal Recommendation**: {waste_info['recommendation']}")
            else:
                st.warning(f"Classification result: '{category}'. This app currently supports classification of plastic soda bottles only.")
        else:
            st.error(f"Error: Unable to process the image. Status code: {response.status_code}")
            st.error(f"Response: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the backend. Please ensure the backend is running and the URL is correct.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
