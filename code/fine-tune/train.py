import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch
from network import DepthToClassificationModel
import copy
from load import CustomDepthDataset, transform
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import os
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data_loader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Load your dataset
input_train_dir = 'datasets/stanford-background-dataset/images/training'
output_train_dir = 'datasets/stanford-background-dataset/foregrounds_bw/training'
input_val_dir = 'datasets/stanford-background-dataset/images/validation'
output_val_dir = 'datasets/stanford-background-dataset/foregrounds_bw/validation'

# Create datasets
train_dataset = CustomDepthDataset(input_dir=input_train_dir, output_dir=output_train_dir, transform=transform)
val_dataset = CustomDepthDataset(input_dir=input_val_dir, output_dir=output_val_dir, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# Load the pre-trained model
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

# Modify the model by removing the head layer and adding binary classifier
model = DepthToClassificationModel(model)

"""
# TEST
input_dir = "inputs/example_dog"
output_dir = "outputs/depth-anything"
filename = "overture-creations-5sI6fQgYIuo.png"
image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
image = Image.open(f"{os.path.join(parent_dir, input_dir)}/{filename}").convert("RGB")
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
img = Image.fromarray(formatted)
img.save(f"{os.path.join(parent_dir, output_dir)}/{filename}")
"""
#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)


# Train and evaluate the model
model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25)
torch.save(model.state_dict(), 'fine_tuned_model.pth')
