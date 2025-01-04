import random
import requests
import numpy as np
import torch
import os
from huggingface_hub import hf_hub_download
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

# Define the disposal recommendations dictionary
disposal_methods = { 
    # Your disposal methods dictionary...
}

# Define a basic dataset structure to handle image loading
class Display:
    @staticmethod
    def show_all(image, anns, dimensions=[8, 16]):
        plt.figure(figsize=(dimensions[0], dimensions[1]))

        # Display the base image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Base Image')
        plt.axis('off')

        # Display the image with the mask
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.title('Image With Mask')
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)

        plt.tight_layout()
        st.pyplot(plt)

# Preprocess image for SAM input (resize long side to 1024 and convert to array)
def preprocess_for_sam(image):
    long_side = 1024
    aspect_ratio = image.width / image.height

    if image.width > image.height:
        new_width = long_side
        new_height = int(long_side / aspect_ratio)
    else:
        new_height = long_side
        new_width = int(long_side * aspect_ratio)

    # Resize the image with the calculated dimensions
    image_resized = image.resize((new_width, new_height))

    # Convert to numpy array
    image_np = np.array(image_resized)

    return image_np

def download_model_from_github(model_url, model_path):
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()  # This will raise an exception for 4xx/5xx errors
        with open(model_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Model downloaded successfully and saved to {model_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the model: {e}")
        raise


# Function to load the SAM model
def load_sam_model():
    model_url = "https://github.com/wahaj4u/AI_Waste_Management_System/releases/download/v1/sam_vit_b.pth"
    model_path = "sam_vit_b.pth"  # Set model_path to the desired path

    if not os.path.exists(model_path):
        print(f"Model not found. Downloading from {model_url}...")
        download_model_from_github(model_url, model_path)
    else:
        print(f"Model found at {model_path}. Using the existing model.")

    # Load the model here
    checkpoint = torch.load(model_path, map_location="cpu")
    sam_model = sam_model_registry["vit_b"](checkpoint)
    sam_model.to(device='cpu')

    mask_generator = SamAutomaticMaskGenerator(sam_model)
    return mask_generator


# Load classification model
def load_classification_model():
    # Load the pre-trained model for waste classification (train_account_best.pth)
    checkpoint = torch.load('train_account_best.pth', weights_only=True)
    model = WasteClassificationModelWithMask(num_classes=len(disposal_methods))  # Adjust with correct number of classes
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class WasteClassificationModelWithMask(torch.nn.Module):
    def __init__(self, num_classes):
        super(WasteClassificationModelWithMask, self).__init__()
        self.backbone = mobilenet_v2(pretrained=True).features

        # Update the first convolution layer to accept 4 channels instead of 3
        self.backbone[0][0] = torch.nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1280, 256),  # Adjust based on MobileNetV2 output channel (1280)
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, num_classes)  # Number of classes
        )

    def forward(self, image_tensor, mask_tensor):
        # Ensure the mask_tensor has the same number of channels as image_tensor
        mask_tensor = torch.nn.Conv2d(mask_tensor.size(1), 1, kernel_size=1)(mask_tensor)  # Reduce mask to 1 channel

        # Resize mask_tensor to match the height and width of image_tensor
        if mask_tensor.shape[2:] != image_tensor.shape[2:]:
            mask_tensor = F.interpolate(mask_tensor, size=image_tensor.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate the image_tensor and mask_tensor along the channel dimension (dim=1)
        concatenated_tensor = torch.cat([image_tensor, mask_tensor], dim=1)

        # Pass the concatenated tensor through the backbone
        x = self.backbone(concatenated_tensor)

        # Apply global average pooling (mean over height and width)
        x = x.mean([2, 3])  # Global average pooling

        # Pass the result through the classifier
        x = self.classifier(x)

        return x

# Streamlit App
def main():
    st.title("WasteSort AI: Waste Sorting and Disposal Assistant")

    # Step 1: Capture Image in Real-Time
    st.subheader("Step 1: Upload an Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Open the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Step 2: Segment the image
        st.subheader("Step 2: Segmenting the Image")
        mask_generator = load_sam_model()

        # Preprocess the image to match SAM input size (long side 1024)
        image_np = preprocess_for_sam(image)

        # Generate segmentation mask
        masks = mask_generator.generate(image_np)

        if masks:
            Display.show_all(image_np, masks)
            mask = masks[0]['segmentation']  # Use the first mask
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            st.image(mask_image, caption="Segmented Mask", use_column_width=True)

            # Step 3: Classify the segmented object
            st.subheader("Step 3: Classifying the Object")

            # Convert the mask to 1 channel
            mask_tensor = ToTensor()(mask_image).unsqueeze(0)

            # Preprocess the original image
            image_tensor = preprocess_for_sam(image)
            image_tensor = torch.tensor(image_tensor).permute(2, 0, 1).unsqueeze(0).float()  # BCHW format

            # Load classification model once
            model = load_classification_model()

            with torch.no_grad():
                # Pass the image and mask tensors separately
                outputs = model(image_tensor, mask_tensor)  # pass both image_tensor and mask_tensor separately
                predicted_class_idx = torch.argmax(outputs, dim=1).item()
                predicted_class = list(disposal_methods.keys())[predicted_class_idx]

            # Step 4: Display disposal recommendation
            st.subheader("Step 4: Disposal Recommendation")
            recommendation = disposal_methods.get(predicted_class, "No recommendation available.")
            st.write(f"**Classified as**: {predicted_class}")
            st.write(f"**Disposal Recommendation**: {recommendation}")
        else:
            st.error("No segmentation mask could be generated.")

if __name__ == "__main__":
    main()
