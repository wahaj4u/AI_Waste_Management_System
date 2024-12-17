import streamlit as st
import requests
from PIL import Image

# Title of the Streamlit App
st.title("Waste Classification App")
st.write("Upload an image to classify the waste and get disposal instructions.")

# File upload section
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "heic"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Send the image to the Flask backend
    st.write("Classifying the waste...")
    files = {"image": uploaded_image.getvalue()}
    
    try:
        # Use the ngrok public URL here
        response = requests.post("https://31ff-34-139-209-102.ngrok-free.app/process", files=files)
        
        if response.status_code == 200:
            result = response.json()
            st.write("### Classification Result:")
            st.write(f"**Category**: {result['category']}")
            st.write(f"**Disposal Recommendation**: {result['disposal_recommendation']}")
            st.write(f"**Type**: {result['type']}")
        else:
            st.error("Error: Unable to classify the image.")
    except Exception as e:
        st.error(f"Failed to connect to backend: {str(e)}")
