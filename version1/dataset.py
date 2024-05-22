#This is the dataset file
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class MVDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
        image = np.array(Image.open(img_path).convert("RGB"),dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # Normalize masks
        mask[mask == 220.0] = 1.0
        # Calculate mean and standard deviation of images
        image = image / 255.0  # Normalize each pixel value to [0, 1]
        mean_value = np.mean(image, axis=(0, 1, 2))  # Calculate mean across each channel
        std_value = np.std(image, axis=(0, 1, 2))  # Calculate std across each channel
        image = (image - mean_value) / std_value  # Normalize using calculated mean and std
        # Normalize images
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        # Normalize images using calculated mean and standard deviation

        return image, mask
