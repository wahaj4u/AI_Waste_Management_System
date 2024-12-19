import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

# Define the model class
class WasteClassificationModel(torch.nn.Module):
    # Define the layers of your model here
    def __init__(self):
        super(WasteClassificationModel, self).__init__()
        # Example layers, adjust according to your actual model architecture
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(128 * 224 * 224, 1024)
        self.fc2 = torch.nn.Linear(1024, 4)  # Assuming 4 classes for classification
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 128 * 224 * 224)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the pre-trained model weights from GitHub
model_url = 'https://github.com/wahaj4u/AI_Waste_Management_System/raw/main/train_account_best.pth'

try:
    # Load model weights from the URL
    state_dict = torch.hub.load_state_dict_from_url(model_url, map_location=torch.device('cpu'))
    
    # Initialize your model
    model = WasteClassificationModel()  # Adjust this if your model has a different initialization method
    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluation mode
except FileNotFoundError:
    st.error("Model file 'train_account_best.pth' not found. Please ensure it is in the correct location.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load the model due to an error: {str(e)}")

# Title of the Streamlit App
st.title("AI Waste Classification Application")
st.write("Capture an image using your camera to classify the waste and get disposal instructions.")

# Disposal recommendation dictionary
disposal_methods = {
    "aerosol_cans": {
        "recommendation": "Make sure the can is empty before disposal. Check with your local recycling program for acceptance. If not recyclable, dispose of as hazardous waste.",
        "type": "Hazardous",
        "recyclable": "Partially Recyclable",
        "compostable": "No"
    },
    "aluminum_food_cans": {
        "recommendation": "Rinse the can thoroughly to remove any food residue. Place it in your recycling bin. Crushing the can saves space but is optional.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    "coffee_grounds": {
        "recommendation": "Coffee grounds are rich in nutrients and can be composted. Add them to your compost bin or garden soil. If composting is not an option, dispose of them in organic waste bins.",
        "type": "Compostable",
        "recyclable": "No",
        "compostable": "Yes"
    },
    "plastic_soda_bottles": {
        "recommendation": "Empty and rinse the bottle before recycling. Leave the cap on if your recycling program accepts it. Crush the bottle to save space if desired.",
        "type": "Recyclable",
        "recyclable": "Yes",
        "compostable": "No"
    },
    # Add additional items as needed
}

# Camera input section
captured_image = st.camera_input("Capture an image using your camera")

if captured_image is not None:
    try:
        # Display the captured image
        image = Image.open(captured_image)
        st.image(image, caption="Captured Image", use_column_width=True)

        # Convert the captured image to the required format for the model
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image).unsqueeze(0)

        # Run the model on the input image
        with torch.no_grad():
            outputs = model(input_tensor)
        
        _, predicted_idx = outputs.max(1)
        categories = ['aerosol_cans', 'aluminum_food_cans', 'coffee_grounds', 'plastic_soda_bottles']  # Extend this list as needed
        category = categories[predicted_idx.item()].lower()

        # Fetch disposal information
        waste_info = disposal_methods.get(category, {
            "recommendation": "No specific instructions available.",
            "type": "Unknown",
            "recyclable": "Unknown",
            "compostable": "Unknown"
        })

        # Display the results
        st.write("### Classification Result:")
        st.write(f"**Category**: {category.capitalize()}")
        st.write(f"**Type**: {waste_info['type']}")
        st.write(f"**Recyclable**: {waste_info['recyclable']}")
        st.write(f"**Compostable**: {waste_info['compostable']}")
        st.write(f"**Disposal Recommendation**: {waste_info['recommendation']}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
