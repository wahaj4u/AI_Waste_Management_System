import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torchvision.models import mobilenet_v2
from torch import nn
import torch.nn.functional as F

# Define the class names and disposal methods
class_names = ['aersosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans',
               'cardboard_boxes', 'cardboard_packaging', 'clothing', 'coffee_grounds',
               'disposable_plastic_cutlery', 'eggshells', 'food_waste', 'glass_beverage_bottles',
               'glass_cosmetic_containers', 'glass_food_jars',	'magazines', 'newspaper', 'office_paper',
               'paper_cups', 'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers',
               'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws', 'plastic_trash_bags',
               'plastic_water_bottles', 'shoes', 'steel_food_cans', 'styrofoam_cups', 'styrofoam_food_containers', 'tea_bags']
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


# Define the WasteClassificationModelWithMask model
class WasteClassificationModelWithMask(nn.Module):
    def __init__(self, num_classes):
        super(WasteClassificationModelWithMask, self).__init__()
        self.backbone = mobilenet_v2(pretrained=True).features

        # Update the first convolution layer to accept 4 channels instead of 3
        self.backbone[0][0] = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256),  # Adjust based on MobileNetV2 output channel (1280)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)  # Number of classes
        )

    def forward(self, image_tensor, mask_tensor):
        # Ensure the mask_tensor has the same number of channels as image_tensor
        mask_tensor = nn.Conv2d(mask_tensor.size(1), 1, kernel_size=1)(mask_tensor)  # Reduce mask to 1 channel

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


# Define the function for classification and mask generation
def classify_with_mask(image, model, mask_generator, device, transform, class_names, disposal_methods):
    # Convert image to numpy array for mask generation
    image_np = np.array(image)
    
    # Generate the mask using SAM
    with torch.no_grad():
        masks = mask_generator.generate(image_np)

    if masks:
        # Use the first mask as the primary mask (example)
        mask = masks[0]['segmentation']
        aggregated_mask = (mask * 255).astype(np.uint8)
    else:
        raise ValueError("No masks were generated for the given image.")

    # Convert aggregated mask to PIL Image and resize
    mask_pil = Image.fromarray(aggregated_mask)
    mask_pil = mask_pil.resize(image.size, Image.NEAREST)

    # Define separate transformations for image and mask
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 3-channel normalization
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()  # Convert mask to tensor
    ])

    # Transform image and mask separately
    image_tensor = image_transform(image)  # RGB image -> 3 channels
    mask_tensor = mask_transform(mask_pil)  # Mask -> 1 channel

    # Concatenate the image and mask as 4 channels
    input_tensor = torch.cat([image_tensor, mask_tensor], dim=0).unsqueeze(0)  # Add batch dimension

    # Perform classification
    model.eval()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # Return the results
    predicted_class_name = class_names[predicted_class]
    disposal_recommendation = disposal_methods[predicted_class_name]
    return predicted_class_name, disposal_recommendation


# Streamlit app
def main():
    st.title("WasteSort AI: Waste Sorting and Disposal Assistant")

    # Step 1: Upload Image
    uploaded_image = st.file_uploader("Upload an image to process", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Step 2: Load model and SAM
        device = torch.device("cpu")
        
        # Load SAM and mask generator
        sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b.pth')
        sam.to(device)
        mask_generator = SamAutomaticMaskGenerator(sam)

        # Load classification model
        model = WasteClassificationModelWithMask(num_classes=len(class_names))
        model.load_state_dict(torch.load('train_loss_best.pt', map_location=device, weights_only=True)['model_state_dict'])
        model.to(device)

        # Step 3: Classify the image and provide recommendations
        predicted_class, recommendation = classify_with_mask(image, model, mask_generator, device, None, class_names, disposal_methods)

        # Display results
        st.subheader("Classification Result")
        st.write(f"**Classified as**: {predicted_class}")
        st.write(f"**Disposal Recommendation**: {recommendation}")


if __name__ == "__main__":
    main()
