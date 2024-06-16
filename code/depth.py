# from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image, ImageFilter
import os, sys
import requests
import cv2

p_d = "/usr/prakt/s0075/image-editing/code/"
# p_d = ""
input_dir = p_d + "inputs/portrait-examples"
output_dir = p_d + "outputs/portrait-examples"


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(parent_dir, 'Depth-Anything'))

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14")

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])



files = os.listdir(input_dir)
for file in files:
    name, _ = os.path.splitext(file)
    print(name)

    image = Image.open(f"{input_dir}/{file}").convert("RGB")
    image = np.array(image) / 255.0
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0)

    # prepare image for the model
    #inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        # outputs = model(inputs)
        formatted = model(image)

        #outputs = model(**inputs)
        #predicted_depth = outputs.predicted_depth

    # interpolate to original size
    #prediction = torch.nn.functional.interpolate(
    #    predicted_depth.unsqueeze(1),
    #    size=image.size[::-1],
    #    mode="bicubic",
    #    align_corners=False,
    #)

    # visualize the prediction
    #output = prediction.squeeze().cpu().numpy()
    #formatted = (output * 255 / np.max(output)).astype("uint8")

    # save background thresholded image
    threshold = 128
    background_mask = formatted > threshold
    background = background_mask.astype(np.uint8)*255
    background_image = Image.fromarray(background)
    background_image.save(f"{output_dir}/{name}-threshold.png")

    # save grayscale depth image
    depth_image = Image.fromarray(formatted)
    depth_image.save(f"{output_dir}/{name}-grayscale.png")

    np_image = np.array(image)
    # blurred_image = cv2.GaussianBlur(np_image, (55, 55), 0)
    # blurred_image = Image.fromarray(blurred_image)
    _, mask_binary = cv2.threshold(formatted, 127, 255, cv2.THRESH_BINARY)
    mask_binary_inv = cv2.bitwise_not(mask_binary)
    clean_roi = cv2.bitwise_and(np_image, np_image, mask=mask_binary)
    clean_roi = cv2.GaussianBlur(clean_roi, (1, 1), 0)

    blurred_roi = cv2.bitwise_and(np_image, np_image, mask=mask_binary_inv)
    blurred_roi = cv2.GaussianBlur(blurred_roi, (155, 155), 0)

    merged_image = cv2.add(blurred_roi, clean_roi)
    merged_image = Image.fromarray(merged_image)
    merged_image.save(f"{output_dir}/{name}-merge.png")

    blurred_image = image.filter(ImageFilter.GaussianBlur(5))
    blurred_image.save(f"{output_dir}/{name}-blur.png")


