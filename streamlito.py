import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from wastesortai_pcode import WasteClassificationModelWithMask, disposal_methods, class_names

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WasteClassificationModelWithMask(num_classes=len(class_names))
model.load_state_dict(torch.load(
    'train_loss_best.pt', map_location=device)['model_state_dict']
)
model.to(device)
model.eval()

# Load the segmentation model
sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b.pth')
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Define transformations
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 3-channel normalization
])
mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Streamlit UI
st.title("Waste Sort AI")
st.write("Upload an image to get waste disposal recommendations.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Processing the image...")

    # Generate a segmentation mask
    image_np = np.array(image)
    with torch.no_grad():
        masks = mask_generator.generate(image_np)

    if masks:
        # Use the first mask (example) and display it
        aggregated_mask = np.sum([mask['segmentation'] for mask in masks], axis=0)
        aggregated_mask = (aggregated_mask > 0).astype(np.uint8) * 255
        mask_image = Image.fromarray(aggregated_mask)
        st.image(mask_image, caption="Segmentation Mask", use_column_width=True)

        # Process the image and mask
        mask_pil = mask_image.resize(image.size, Image.NEAREST)
        image_tensor = image_transform(image).unsqueeze(0)
        mask_tensor = mask_transform(mask_pil).unsqueeze(0)
        input_tensor = torch.cat([image_tensor, mask_tensor], dim=1).to(device)

        # Classify the input
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Get disposal recommendation
        predicted_label = class_names[predicted_class]
        disposal_recommendation = disposal_methods.get(predicted_label, "No recommendation available.")

        # Display results
        st.write(f"**Predicted Class:** {predicted_label}")
        st.write(f"**Disposal Recommendation:** {disposal_recommendation}")
    else:
        st.error("No segmentation mask could be generated for the image.")
