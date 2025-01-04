import streamlit as st
import torch
from PIL import Image
from transformers import SamProcessor, SamModel
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision.models import mobilenet_v2
import torch.nn.functional as F

# Define the disposal recommendations dictionary
disposal_methods = { 
    "aerosol_cans": "Make sure the can is empty before disposal. Check with your local recycling program for acceptance. If not recyclable, dispose of as hazardous waste.",
    "aluminum_food_cans": "Rinse the can thoroughly to remove any food residue. Place it in your recycling bin. Crushing the can saves space but is optional.",
    "aluminum_soda_cans": "Rinse to remove sticky residue. Place the can in your recycling bin. Avoid crushing if your recycling program requires intact cans.",
    "cardboard_boxes": "Flatten the box to save space before recycling. Remove any non-cardboard elements like tape or labels. Place in the recycling bin for paper/cardboard.",
    "cardboard_packaging": "Ensure all packaging is flattened for easy recycling. Remove non-cardboard parts such as plastic film or foam. Recycle with other cardboard materials.",
    "clothing": "If still wearable, consider donating to local charities or thrift stores. For damaged clothing, recycle as fabric or take to textile recycling bins. Avoid placing in general waste.",
    "coffee_grounds": "Coffee grounds are rich in nutrients and can be composted. Add them to your compost bin or garden soil. If composting is not an option, dispose of them in organic waste bins.",
    "disposable_plastic_cutlery": "Most disposable cutlery is not recyclable. Place it in the general waste bin. Consider switching to reusable or compostable alternatives in the future.",
    "eggshells": "Eggshells can be composted and are great for enriching soil. Add them to your compost bin after rinsing. Alternatively, place in organic waste bins.",
    "food_waste": "Separate food waste from packaging before disposal. Compost if possible to reduce landfill impact. Use organic waste bins where available.",
    "glass_beverage_bottles": "Rinse thoroughly to remove any liquid. Place in the glass recycling bin. Remove caps or lids if not made of glass.",
    "glass_cosmetic_containers": "Clean the container to ensure it's residue-free. Recycle if your local program accepts glass containers. Broken glass should be wrapped in paper or cardboard and placed in general waste.",
    "glass_food_jars": "Rinse the jar to remove food residue. Recycle in glass bins. Lids made of metal can often be recycled separately.",
    "magazines": "Remove plastic covers or non-paper elements before recycling. Place in your paper recycling bin. Avoid recycling if excessively wet or damaged.",
    "newspaper": "Keep newspapers dry and free of contaminants like food stains. Recycle them in designated paper bins. Bundle them for easier handling if required.",
    "office_paper": "Shred confidential documents if necessary before recycling. Avoid including paper with heavy lamination or plastic content. Recycle in paper bins.",
    "paper_cups": "Check for a recycling symbol to confirm if recyclable. Most paper cups with plastic lining are not recyclable and go into general waste. Consider switching to reusable cups.",
    "plastic_cup_lids": "If marked recyclable, clean and place them in the appropriate bin. Otherwise, dispose of in general waste. Avoid using single-use lids when possible.",
    "plastic_detergent_bottles": "Rinse out any remaining detergent to avoid contamination. Check the recycling symbol and place in plastics recycling. Keep the lid on if acceptable.",
    "plastic_food_containers": "Ensure the container is clean and free of food residue. Recycle if marked as recyclable. Otherwise, dispose of in general waste.",
    "plastic_shopping_bags": "Reuse them for storage or garbage liners. If recycling facilities for plastic bags are available, drop them off. Avoid throwing in general recycling bins.",
    "plastic_soda_bottles": "Empty and rinse the bottle before recycling. Leave the cap on if your recycling program accepts it. Crush the bottle to save space if desired.",
    "plastic_straws": "Plastic straws are not recyclable in most programs. Dispose of them in general waste. Consider using reusable or biodegradable straws.",
    "plastic_trash_bags": "Trash bags themselves are not recyclable. Dispose of them in general waste along with their contents. Look for biodegradable options when purchasing new ones.",
    "plastic_water_bottles": "Rinse the bottle to ensure cleanliness. Recycle the bottle along with the cap if accepted. Try to use reusable bottles to reduce plastic waste.",
    "shoes": "Donate shoes that are still wearable to charities or thrift stores. For damaged or unusable shoes, check for textile recycling bins. Avoid discarding in general waste.",
    "steel_food_cans": "Clean the can by removing all food residue. Place it in your recycling bin. Check for local recycling guidelines if needed.",
    "styrofoam_cups": "Styrofoam is not recyclable in most areas. Dispose of it in general waste. Avoid using Styrofoam products whenever possible.",
    "styrofoam_food_containers": "Clean the container before disposal if required. Place it in general waste as Styrofoam is typically non-recyclable. Consider switching to sustainable alternatives.",
    "tea_bags": "Compost biodegradable tea bags as they are rich in organic matter. Check if your tea bags have plastic components and dispose of those in general waste."
}

# Load the SAM model directly from Hugging Face
@st.cache_resource
def load_model():
    model_name = "Wahaj4u/sam_vit_b"  
    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name)
    return processor, model

# Function to process the uploaded image
def segment_image(image):
    processor, model = load_model()

    # Convert the uploaded image to the format the model expects
    inputs = processor(images=image, return_tensors="pt")

    # Perform segmentation
    with torch.no_grad():
        outputs = model(**inputs)

    segmentation_mask = outputs.logits.argmax(dim=1)  
    return segmentation_mask

# Display Segmentation Results
def display_segmentation_results(image, segmentation_mask):
    plt.figure(figsize=(8, 8))
    
    # Display the base image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Base Image')
    plt.axis('off')
    
    # Display the segmentation mask
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_mask[0], cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    st.pyplot(plt)

# Preprocess image for SAM input (resize long side to 1024 and convert to array)
def preprocess_for_sam(image):
    # Resize the image such that the long side is 1024 while preserving the aspect ratio
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

# Define the classification model for waste classification (simplified)
def load_classification_model():
    checkpoint = torch.load('train_account_best.pth')
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
    st.subheader("Step 1: Capture an Image")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Step 2: Segmenting the Image
        st.subheader("Step 2: Segmenting the Image")
        
        # Use the load_model function to load processor and model
        processor, model = load_model()

        # Preprocess the image to match SAM input size (long side 1024)
        image_np = preprocess_for_sam(image)

        # Generate segmentation mask using the processor and model
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        segmentation_mask = outputs.logits.argmax(dim=1)  # Get the segmentation mask
        
        # Display the segmentation results
        display_segmentation_results(image, segmentation_mask)

        # Step 3: Classify the Object
        st.subheader("Step 3: Classifying the Object")

        # Convert the mask to 1 channel
        mask_tensor = ToTensor()(segmentation_mask[0].cpu().numpy()).unsqueeze(0)

        # Preprocess the original image
        image_tensor = np.array(image)
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
