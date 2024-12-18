import streamlit as st
import requests
from PIL import Image

# Title of the Streamlit App
st.title("Waste Classification App")
st.write("Capture an image using your camera to classify the waste and get disposal instructions.")

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
        backend_url = "https://ffd8-35-239-193-174.ngrok-free.app/process"

        # Send the image to the Flask backend
        response = requests.post(backend_url, files=files)

        if response.status_code == 200:
            result = response.json()
            st.write("### Classification Result:")
            st.write(f"**Category**: {result.get('category', 'N/A')}")
            st.write(f"**Disposal Recommendation**: {result.get('disposal_recommendation', 'N/A')}")
            st.write(f"**Type**: {result.get('type', 'N/A')}")
        else:
            st.error(f"Error: Unable to classify the image. Status code: {response.status_code}")
            st.error(f"Response: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the backend. Please ensure the backend is running and the URL is correct.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
