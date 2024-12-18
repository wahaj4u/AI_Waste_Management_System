import streamlit as st
import requests
from PIL import Image

# Title of the Streamlit App
st.title("Waste Classification App")
st.write("Upload an image to classify the waste and get disposal instructions.")

# File upload section
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "heic"])

if uploaded_image is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the uploaded file to bytes for the POST request
        st.write("Classifying the waste...")
        files = {"image": uploaded_image.getvalue()}
        
        # Replace with the active ngrok or backend URL
        backend_url = "https://bd17-35-225-208-157.ngrok-free.app/process"

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
