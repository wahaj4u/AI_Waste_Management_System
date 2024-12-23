# General utilities
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# PyTorch and torchvision for deep learning
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional
from torchvision import models, transforms
from torchvision.transforms import v2
from torchvision.models import mobilenet_v2, mobilenet_v3_small
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

# Segment Anything Model (SAM) for segmentation
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Scikit-learn for evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

"""# **Section 2: Dataset Preparation and Fine-tuning**"""

# Initialize Datasets
dataset_path = "/content/drive/MyDrive/WASTESORTAI/images"
output_path = "/content/drive/MyDrive/WASTESORTAI/output_masks"

root_dir = "/content/drive/MyDrive/WASTESORTAI/images"
mask_dir = "/content/drive/MyDrive/WASTESORTAI/output_masks"

# PreProcess Image Data Transformations
train_pil_transform = v2.Compose([
        v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2),
        v2.RandomAffine(degrees=5, translate=(0.1, 0.1),scale=(0.8,1.3),interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        v2.Resize(size=(256, 256)),
        v2.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
    ])

val_pil_transform = v2.Compose([
    v2.Resize(size=(256, 256)),
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


data_transforms = {
    "train":train_pil_transform,
    "val":val_pil_transform,
    "test":test_pil_transform,
}

#Prepare Custom Dataset Class
class WasteDatasetWithMasks(Dataset):
    def __init__(self, root_dir, mask_dir, split='train', transform=None, dirs_select_from=['default', 'real_world']):
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.split = split
        self.transform = transform
        self.dirs_select_from = dirs_select_from
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.mask_paths = []
        self.labels = []

        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            mask_class_dir = os.path.join(mask_dir, class_name)

            for subfolder in self.dirs_select_from:
                subfolder_dir = os.path.join(class_dir, subfolder)
                mask_subfolder_dir = os.path.join(mask_class_dir, subfolder)

                if not os.path.exists(subfolder_dir) or not os.path.exists(mask_subfolder_dir):
                    continue

                image_names = os.listdir(subfolder_dir)
                random.shuffle(image_names)

                if split == 'train':
                    image_names = image_names[:int(0.7 * len(image_names))]
                elif split == 'val':
                    image_names = image_names[int(0.7 * len(image_names)):int(0.9 * len(image_names))]
                elif split == 'test':
                    image_names = image_names[int(0.9 * len(image_names)):]

                for image_name in image_names:
                    image_path = os.path.join(subfolder_dir, image_name)
                    mask_path = os.path.join(mask_subfolder_dir, f"{os.path.splitext(image_name)[0]}_mask.png")

                    if os.path.exists(mask_path):  # Ensure the mask exists
                        self.image_paths.append(image_path)
                        self.mask_paths.append(mask_path)
                        self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Load mask as grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Stack the mask as an additional channel
        combined_input = torch.cat([image, mask.unsqueeze(0)], dim=0)

        return {"image": combined_input, "label": label}

train_dataset = WasteDataset("/content/drive/MyDrive/WASTESORTAI/images","train",data_transforms["train"])
val_dataset = WasteDataset("/content/drive/MyDrive/WASTESORTAI/images","val",data_transforms["val"])
test_dataset = WasteDataset("/content/drive/MyDrive/WASTESORTAI/images", "",data_transforms["test"])

image_datasets = {
    "train":train_dataset,
    "val":val_dataset,
    "test":test_dataset
}

class_names = train_dataset.classes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
batch_size = 16

train_data_loader = DataLoader(train_dataset,batch_size,True, num_workers=int(os.cpu_count()*0.8))
val_data_loader = DataLoader(val_dataset, batch_size, False, num_workers=int(os.cpu_count()*0.2))
test_data_loader = DataLoader(test_dataset, batch_size, True)

data_loaders = {
    "train":train_data_loader,
    "val":val_data_loader,
    "test":test_data_loader
}

"""# **Section 3: SAM2 Model Segmentation**

"""

# Ensure that PyTorch is available and a GPU is accessible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")  # Print the device being used (cuda or cpu)

# Load SAM2 model configuration
config = {
    'MODEL_TYPE': 'vit_b',
    'SAM_CHECKPOINT': '/content/drive/MyDrive/WASTESORTAI/sam_vit_b.pth',
    'device': 'cuda', # Currently using GPU
}

# Load the SAM2 model
sam = sam_model_registry[config['MODEL_TYPE']](checkpoint=config['SAM_CHECKPOINT'])
sam.to(device=config['device'])
mask_generator = SamAutomaticMaskGenerator(sam)

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Function to process images in a folder and save corresponding masks
def process_images_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    print(f"Processing folder: {input_folder}")  # Debug: Check the folder being processed

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Process only image files
            print(f"Processing image: {file_name}")  # Debug: Check which image is being processed

            # Read image
            image = cv2.imread(file_path)
            if image is None:
                print(f"Failed to read image: {file_name}")  # Debug: Image reading failed
                continue

            # Convert to RGB (as SAM expects RGB format)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate masks using SAM2
            with torch.no_grad():
                masks = mask_generator.generate(image_rgb)  # Generate masks for the image

            # Aggregate masks into one mask (sum over masks)
            if masks:
                # Create a sum of all masks' segmentation
                aggregated_mask = np.sum([mask['segmentation'] for mask in masks], axis=0)
                # Convert to binary mask
                aggregated_mask = (aggregated_mask > 0).astype(np.uint8) * 255

                # Output path for saving the mask
                output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_mask.png")
                print(f"Saving aggregated mask to: {output_file}")

                # Save the aggregated mask as a PNG file
                result = cv2.imwrite(output_file, aggregated_mask)

                if result:
                    print(f"Mask saved: {output_file}")  # Debug: Confirm the mask is saved
                else:
                    print(f"Failed to save mask: {output_file}")  # Debug: Error saving mask
            else:
                print(f"No masks generated for image: {file_name}")  # Debug: No masks generated

# Iterate through dataset folders and process images
print(f"Dataset path: {dataset_path}")  # Debug: Print the dataset path to check
for main_dir in os.listdir(dataset_path):
    main_dir_path = os.path.join(dataset_path, main_dir)

    if os.path.isdir(main_dir_path):
        print(f"Processing main directory: {main_dir}")  # Debug: Check main directory

        # Process both 'default' and 'real_world' subfolders
        for sub_dir in ['default', 'real_world']:
            sub_dir_path = os.path.join(main_dir_path, sub_dir)

            # Check if sub_dir exists in the current main_dir
            if os.path.isdir(sub_dir_path):
                print(f"Processing subdirectory: {sub_dir}")  # Debug: Check subdirectory

                # Create corresponding output path
                sub_dir_output_path = os.path.join(output_path, main_dir, sub_dir)
                os.makedirs(sub_dir_output_path, exist_ok=True)  # Ensure the subdirectory is created

                # Process images and save masks
                process_images_in_folder(sub_dir_path, sub_dir_output_path)
            else:
                print(f"Subdirectory {sub_dir} does not exist in {main_dir}")  # Debug: Subdirectory not found

print("Processing complete. Masks saved in the output directory.")

"""# **Section 4: Data Visualization**"""

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the visualize_batch function
def visualize_batch(batch, classes, dataset_type):
    # initialize a figure
    fig = plt.figure("{} batch".format(dataset_type),
                     figsize=(batch_size, batch_size))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    # loop over the batch size
    for i in range(0, batch_size):
        # create a subplot with 4x8
        ax = plt.subplot(4, 4, i + 1)
        # grab the image, convert it from channels first ordering to
        # channels last ordering, and scale the raw pixel intensities
        # to the range [0, 255]
        image = batch["image"][i].cpu().numpy()
        image = image.transpose((1, 2, 0))
        image = std * image + mean
        image = image.astype("uint8")
        # grab the label id and get the label from the classes list
        idx = batch["label"][i]
        label = classes[idx]
        # show the image along with the label
        plt.imshow(image)
        plt.title(label)
        plt.axis("off")
    # show the plot
    plt.tight_layout()
    plt.show()

#visualize train data
train_batch = next(iter(data_loaders["train"]))
visualize_batch(train_batch,class_names,"train")

"""# **Section 5: Integration with MobileNetV2 Classifier**"""

# Disposal dictionary
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

# Create class_names from the disposal dictionary keys
class_names = list(disposal_methods.keys())

# Define the model
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

    def forward(self, x):
        x = self.backbone(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x

# Initialize the model and move it to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WasteClassificationModel()
model = model.to(device)

# Verify the device the model is on
print(f"Model is on device: {next(model.parameters()).device}")
print(f"Number of classes: {len(class_names)}")

"""# **Section 6: Model Training**"""

# Define the dataset and transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
])

mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match image size
    transforms.ToTensor(),  # Convert mask to tensor
])

"""# **Section 5: Model Initialization & Training**"""

# Assuming WasteClassificationModel is defined
model = WasteClassificationModel()  # Replace with your actual model class
model = model.to(device)
criterion = nn.CrossEntropyLoss()
model_optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Number of epochs
num_epochs = 30
best_acc = {"train": -1000, "val": -1000}
best_loss = {"train": 1000, "val": 1000}

# Paths for saving the best models
base_path = '/content/drive/MyDrive/WASTESORTAI/model'
best_accuracy_model_path = {
    "train": os.path.join(base_path, "train_acc_best.pt"),
    "val": os.path.join(base_path, "val_acc_best.pt")
}
best_loss_model_path = {
    "train": os.path.join(base_path, "train_loss_best.pt"),
    "val": os.path.join(base_path, "val_loss_best.pt")
}

# Create lists to store loss and accuracy for each epoch
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

# Start training loop
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0.0

        for idx, data in enumerate(data_loaders[phase]):
            inputs, labels = data["image"].to(device), data["label"].to(device)
            model_optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    model_optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Store loss and accuracy for plotting
        if phase == "train":
            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc.cpu())  # Move to CPU before appending
        else:
            val_loss_history.append(epoch_loss)
            val_acc_history.append(epoch_acc.cpu())  # Move to CPU before appending

        # Save best model based on accuracy or loss
        if epoch_acc > best_acc[phase]:
            best_acc[phase] = epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_optimizer.state_dict(),
                'loss': epoch_loss
            }, best_accuracy_model_path[phase])

        if epoch_loss < best_loss[phase]:
            best_loss[phase] = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_optimizer.state_dict(),
                'loss': epoch_loss
            }, best_loss_model_path[phase])

# After training, plot the loss and accuracy curves

# Plot Loss Curve
plt.figure(figsize=(12, 6))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), train_loss_history, label='Training Loss', color='blue')
plt.plot(range(num_epochs), val_loss_history, label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), train_acc_history, label='Training Accuracy', color='blue')
plt.plot(range(num_epochs), val_acc_history, label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()

# Define image transformations for your trained model
cropped_images = []
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),  # Resize to 256x256
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

# Update the model's fc layer to match your number of classes
num_classes = len(disposal_methods)  # Number of output classes
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# Classify each cropped object
for idx, cropped in enumerate(cropped_images):
    if len(cropped.shape) == 3:  # Ensure cropped image is in correct format
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if using OpenCV

    input_tensor = transform(cropped).unsqueeze(0).to(device)  # Add batch dimension
    print(f"Input tensor shape: {input_tensor.shape}")  # Debugging

    with torch.no_grad():
        outputs = model(input_tensor)

    # Get the predicted class
    predicted_class = torch.argmax(outputs, dim=1).item()

    # Fetch class name and disposal instructions
    class_name, disposal_instruction = disposal_methods.get(predicted_class, ("Unknown", "No instructions available"))
    print(f"Object {idx}:")
    print(f"  Predicted Class: {class_name}")
    print(f"  Disposal Instructions: {disposal_instruction}")

"""# **Section 6: Model Evaluation**"""

model = WasteClassificationModel()
checkpoint = torch.load("/content/drive/MyDrive/WASTESORTAI/model/train_loss_best.pt")
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

data_size = len(image_datasets["val"])
y_preds = []
y_true = []

for idx, data in enumerate(data_loaders["val"]):
    inputs, labels = data["image"].to(device), data["label"].to(device)

    outputs = model(inputs)
    _, predictions = torch.max(outputs,1)
    y_preds.extend(predictions.cpu().numpy().tolist())
    y_true.extend(labels.cpu().numpy().tolist())

print("\n-------Classification Report-------\n")
print(classification_report(y_true, y_preds,target_names=class_names))
cm = confusion_matrix(y_true, y_preds)
cm_disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
fig, ax = plt.subplots(figsize=(16,14))
cm_disp.plot(ax=ax)

"""# **Section 7: Disposal Recommendations**"""

test_batch = next(iter(data_loaders["test"]))
inputs, labels = test_batch["image"].to(device), test_batch["label"].to(device)

outputs = model(inputs)
_, predictions = torch.max(outputs.data,1)


fig = plt.figure("Test Batch with Recommendations", figsize=(10, 10))
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])



# Get a batch of test data
test_batch = next(iter(data_loaders["test"]))
inputs, labels = test_batch["image"].to(device), test_batch["label"].to(device)

# Model predictions
outputs = model(inputs)
_, predictions = torch.max(outputs.data, 1)

# Create a figure to plot the images
fig = plt.figure("Test Batch", figsize=(16, 16))

# Loop through the batch
for i in range(len(predictions)):
    ax = plt.subplot(4, 4, i + 1)  # Adjust grid size as needed

    # Denormalize the image
    image = inputs[i].cpu().numpy()
    image = image.transpose((1, 2, 0))  # Convert to HWC format
    image = std * image + mean  # Reverse normalization
    image = np.clip(image, 0, 1)  # Clip pixel values to valid range

    # Predicted label and disposal recommendation
    predicted_label = class_names[predictions[i]]
    disposal_recommendation = disposal_methods.get(predicted_label, "No recommendation available.")

    # Display the image
    plt.imshow(image)
    idx = predictions[i].item()
    label = class_names[idx]
    true_label = class_names[labels[i]]
    title = f"True: {true_label}\nPred: {predicted_label}\nDispose: {disposal_recommendation}"
    plt.title(title, fontsize=8)
    plt.axis("off")

# Show the plot
plt.tight_layout()
plt.show()

#Print disposal recommendations in console
for i in range(len(predictions)):
    predicted_label = class_names[predictions[i]]
    disposal_recommendation = disposal_methods.get(predicted_label, "No recommendation available.")
    print(f"Item: {predicted_label}")
    print(f"Disposal Recommendation: {disposal_recommendation}\n")

def classify_with_mask(image_path, model, mask_generator, device, transform, class_names, disposal_methods):
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be loaded.")

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate the mask using SAM
    with torch.no_grad():
        masks = mask_generator.generate(image_rgb)

    if masks:
        # Combine masks to create a single binary mask
        aggregated_mask = np.sum([mask['segmentation'] for mask in masks], axis=0)
        aggregated_mask = (aggregated_mask > 0).astype(np.uint8) * 255
    else:
        raise ValueError("No mask generated for the image.")

    # Preprocess the image and mask
    image_pil = Image.fromarray(image_rgb)
    mask_pil = Image.fromarray(aggregated_mask)

    image_tensor = transform(image_pil)
    mask_tensor = transform(mask_pil)

    # Stack image and mask
    combined_input = torch.cat([image_tensor, mask_tensor.unsqueeze(0)], dim=0).unsqueeze(0).to(device)

    # Classify the object
    model.eval()
    with torch.no_grad():
        outputs = model(combined_input)
        predicted_idx = torch.argmax(outputs, dim=1).item()

    # Get class name and disposal recommendation
    class_name = class_names[predicted_idx]
    disposal_recommendation = disposal_methods.get(class_name, "No recommendation available.")

    print(f"Classified as: {class_name}")
    print(f"Disposal Recommendation: {disposal_recommendation}")

    return class_name, disposal_recommendation

# Initialize SAM2 and WasteClassificationModelWithMask
sam = sam_model_registry['vit_b'](checkpoint='/content/drive/MyDrive/WASTESORTAI/sam_vit_b.pth')
sam.to(device)

model = WasteClassificationModelWithMask(num_classes=len(class_names))
model.load_state_dict(torch.load('/content/drive/MyDrive/WASTESORTAI/model/train_loss_best.pt')['model_state_dict'])
model.to(device)

# Transform for preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 4, std=[0.5] * 4)
])

# Real-time prediction
image_path = "path_to_image.jpg"
classify_with_mask(image_path, model, sam, device, transform, class_names, disposal_methods)
