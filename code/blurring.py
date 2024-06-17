import torch
import numpy as np
from PIL import Image, ImageFilter
import os, sys
import cv2


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


def apply_blur(input_dir: str, output_dir: str):
    files = os.listdir(input_dir)
    for file in files:
        name, _ = os.path.splitext(file)
        print(name)

        image = Image.open(f"{input_dir}/{file}").convert("RGB")
        image = np.array(image)
        image_norm = np.array(image) / 255.0
        image_norm = transform({'image': image_norm})['image']
        image_norm = torch.from_numpy(image_norm).unsqueeze(0)

        with torch.no_grad():
            prediction = model(image_norm)

        # visualize the prediction
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        np_image = np.array(image_norm).squeeze(0).transpose(1, 2, 0).astype(np.uint8)
        resized_image = cv2.resize(image, (np_image.shape[1], np_image.shape[0]))

        # blur image given depth map
        sharpen = 0
        boxBlur = 5

        oimg = Image.fromarray(resized_image)
        mimg = Image.fromarray(formatted)

        bimg = oimg.filter(ImageFilter.BoxBlur(int(boxBlur)))
        bimg = bimg.filter(ImageFilter.BLUR)
        for i in range(sharpen):
            bimg = bimg.filter(ImageFilter.SHARPEN)

        rimg = Image.composite(oimg, bimg, mimg)
        rimg.save(f"{output_dir}/{name}-blur.png")

p_d = "/usr/prakt/s0075/image-editing/code/"
input_dir = p_d + "inputs/portrait-examples"
output_dir = p_d + "outputs/portrait-examples"
apply_blur(input_dir, output_dir)