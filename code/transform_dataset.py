from PIL import Image
import numpy as np
import os
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch


def masks_to_bw():
    # Open the image
    input_image_path = 'datasets/stanford-background-dataset/labels_colored'
    output_image_path = 'datasets/stanford-background-dataset/foregrounds_bw'

    files = os.listdir(input_image_path)

    for file in files:
        image = Image.open(f"{input_image_path}/{file}").convert("RGB")

        # Convert the image to an array
        image_array = np.array(image)

        # Define the target color
        target_color = np.array([180, 222, 44])

        # Create a mask where the target color is transformed to white and the rest to black
        mask = np.all(image_array == target_color, axis=-1)
        # grayscale_array = np.zeros(image_array.shape[:2], dtype=np.uint8)
        # grayscale_array[mask] = 255
        foreground_bw = mask.astype(np.uint8) * 255

        # Convert the array back to a PIL grayscale image
        foreground_bw_image = Image.fromarray(foreground_bw)

        # Save or display the grayscale image
        foreground_bw_image.save(f"{output_image_path}/{file}")


def color_to_depth():
    input_dir = "datasets/stanford-background-dataset/images"
    output_dir = "datasets/stanford-background-dataset/depth_images"

    image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

    files = os.listdir(input_dir)
    for file in files:

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

        # save grayscale depth image
        depth_image = Image.fromarray(formatted)
        depth_image.save(f"{output_dir}/{file}")


color_to_depth()
