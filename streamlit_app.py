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

        # Send the image to the Flask backend
        st.write("Classifying the waste...")
        files = {"image": uploaded_image.getvalue()}  # Get raw bytes of the uploaded file

        # Replace with your backend URL
        flask_url = "http://127.0.0.1:5000/process"  

        # Make a POST request to the Flask backend
        response = requests.post(flask_url, files=files)

        # Handle the response
        if response.status_code == 200:
            result = response.json()
            st.write("### Classification Result:")
            st.write(f"**Category**: {result['category']}")
            st.write(f"**Disposal Recommendation**: {result['disposal_recommendation']}")
            st.write(f"**Type**: {result['type']}")
        elif response.status_code == 400:
            st.error("Error: Bad Request. Ensure the image is sent correctly.")
        elif response.status_code == 500:
            st.error("Error: Internal Server Error. Check backend logs.")
        else:
            st.error(f"Unexpected Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Error: Unable to connect to the backend. Ensure Flask is running.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
