import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as F
import sys
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

class CustomDepthDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
        self.input_images = os.listdir(os.path.join(parent_dir, input_dir))
        self.output_images = os.listdir(os.path.join(parent_dir, output_dir))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        output_image_path = os.path.join(self.output_dir, self.output_images[idx])

        input_image = Image.open(os.path.join(parent_dir, input_image_path)).convert('RGB')
        output_image = Image.open(os.path.join(parent_dir, output_image_path))# .convert('L')  # Grayscale

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image


class ToTensorWithMask(object):
    def __call__(self, sample):
        input_image, output_image = sample
        input_image = F.to_tensor(input_image)
        output_image = torch.tensor(np.array(output_image), dtype=torch.long)
        return input_image, output_image

"""
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    ToTensorWithMask(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
"""

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to torch tensor
    # Add more transforms here if needed
])
