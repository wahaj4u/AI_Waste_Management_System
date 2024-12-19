import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import mobilenet_v3_small

# Custom Dataset Class
class WasteDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
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

                if split == 'train':
                    image_names = image_names[:int(0.6 * len(image_names))]
                elif split == 'val':
                    image_names = image_names[int(0.6 * len(image_names)):int(0.8 * len(image_names))]
                else:  # split == 'test'
                    image_names = image_names[int(0.8 * len(image_names)):]

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

# Prepare datasets and dataloaders
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

train_dataset = WasteDataset("/kaggle/input/recyclable-and-household-waste-classification/images/images", "train",
                             data_transforms["train"])
val_dataset = WasteDataset("/kaggle/input/recyclable-and-household-waste-classification/images/images", "val",
                           data_transforms["val"])
test_dataset = WasteDataset("/kaggle/input/recyclable-and-household-waste-classification/images/images", "test",
                            data_transforms["test"])

image_datasets = {
    "train":train_dataset,
    "val":val_dataset,
    "test":test_dataset
}

class_names = train_dataset.classes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
batch_size = 16

train_data_loader = DataLoader(train_dataset, batch_size, True, num_workers=int(os.cpu_count() * 0.8))
val_data_loader = DataLoader(val_dataset, batch_size, False, num_workers=int(os.cpu_count() * 0.2))
test_data_loader = DataLoader(test_dataset, batch_size, True, num_workers=int(os.cpu_count() * 0.2))

data_loaders = {
    "train":train_data_loader,
    "val":val_data_loader,
    "test":test_data_loader
}

# Waste Classification Model
class WasteClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobnet = mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small
