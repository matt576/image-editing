from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import os
import requests
from sklearn.cluster import KMeans

output_dir = "outputs/depth-anything"
input_dir = "inputs/examples_depth"
# filename = "cats-depth"

image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

files = os.listdir(input_dir)
for file in files:
    name, _ = os.path.splitext(file)
    print(name)

    image = Image.open(f"{input_dir}/{file}").convert("RGB")

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

    threshold = 128  # Adjust this threshold as needed
    background_mask = formatted > threshold
    background = background_mask.astype(np.uint8)*255

    # save background thresholded image
    background_image = Image.fromarray(background)
    background_image.save(f"{output_dir}/{name}-threshold.png")

    # save grayscale depth image
    depth_image = Image.fromarray(formatted)
    depth_image.save(f"{output_dir}/{name}-grayscale.png")

    # Reshape depth map for K-means clustering
    depth_flat = formatted.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(depth_flat)
    labels = kmeans.labels_.reshape(formatted.shape)
    background_label = labels.mean() > 0.5
    background_cluster_mask = (labels == background_label)
    background_cluster = background_cluster_mask.astype(np.uint8)*255
    background_cluster_image = Image.fromarray(background)
    background_cluster_image.save(f"{output_dir}/{name}-cluster.png")
