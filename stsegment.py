import streamlit as st
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import matplotlib.pyplot as plt

# Configuration for SAM
config = {
    'MODEL_TYPE': 'vit_b',
    'SAM_CHECKPOINT': 'sam_vit_b.pth',
    'device': 'cpu'
}

# Initialize the SAM model
sam = sam_model_registry[config['MODEL_TYPE']](checkpoint=config['SAM_CHECKPOINT'])
sam.to(device=config['device'])
mask_generator = SamAutomaticMaskGenerator(sam)

# Streamlit UI
st.title("Waste Segmentation App")
st.write("Upload an image and the SAM model will segment objects automatically.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

def display_masks(image, masks):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title("Segmented Image")
    sorted_anns = sorted(masks, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    st.pyplot(plt)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Segment using SAM
    st.write("Processing the image...")
    masks = mask_generator.generate(image_np)
    display_masks(image_np, masks)
else:
    st.write("Please upload an image to continue.")
