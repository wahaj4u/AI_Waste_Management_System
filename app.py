import streamlit as st
import torch
import torchvision.transforms as v2
from PIL import Image
from model import WasteClassificationModel

# Load the model
def load_model(model_url):
    try:
        model_weights = torch.hub.load_state_dict_from_url(model_url, map_location=torch.device('cpu'))
        return model_weights
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None

# Define the Waste Dataset
class WasteDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []

        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for subfolder in ['default', 'real_world']:
                subfolder_dir = os.path.join(class_dir, subfolder)
                image_names = os.listdir(subfolder_dir)
                random.shuffle(image_names)

                for image_name in image_names:
                    self.image_paths.append(os.path.join(subfolder_dir, image_name))
                    self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        data = {
            "image":image,
            "label":label
        }
        return data

def main():
    st.title("Waste Classification App")

    # Load the model
    model_url = "zzzz"
    model_weights = load_model(model_url)
    if model_weights is None:
        return

    model = WasteClassificationModel()
    model.load_state_dict(model_weights)
    model.eval()  # Set model to evaluation mode

    # Transform definitions
    train_pil_transform = v2.Compose([
        v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2),
        v2.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.8, 1.3),
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        v2.Resize(size=(256, 256)),
        v2.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    test_pil_transform = v2.Compose([
        v2.Resize(size=(256, 256)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Application logic
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = test_pil_transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            class_names = model.classes
            prediction = class_names[preds[0].item()]
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
