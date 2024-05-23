from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import os

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = Image.open(requess.get(url, stream=True).raw)

image_path = "inputs/phone.png"
full_image_path = os.path.abspath(image_path)
print("Full path:", full_image_path)
image = Image.open(image_path)

model_name = "depth-anything-small-hf"
image_processor = AutoImageProcessor.from_pretrained(f"LiheYoung/{model_name}")
model = AutoModelForDepthEstimation.from_pretrained(f"LiheYoung/{model_name}")

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# visualize the prediction
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)

depth.save(f"outputs/{model_name}/phone.png")
