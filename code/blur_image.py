import torch
import numpy as np
from PIL import Image, ImageFilter
import os, sys
import cv2
from mask_foreground import extract_foreground, scale_and_paste


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



def apply_blur(input_img: Image, foreground_img: Image, boxBlur: int, sharpen: int):

    image = np.array(input_img)
    image_norm = np.array(image) / 255.0
    image_norm = transform({'image': image_norm})['image']
    image_norm = torch.from_numpy(image_norm).unsqueeze(0)
    
    # resize the orignal image to the normalized
    np_image = np.array(image_norm).squeeze(0).transpose(1, 2, 0).astype(np.uint8)
    resized_image = cv2.resize(image, (np_image.shape[1], np_image.shape[0]))

    # run depth anything
    with torch.no_grad():
        output = model(image_norm)

    # visualize the prediction
    output = output.squeeze().cpu().numpy()
    prediction = (output * 255 / np.max(output)).astype("uint8")

    """
    # put the foreground mask on top of the depth map
    foreground_img = np.array(foreground_img)
    resized_foreground_img = cv2.resize(foreground_img, (np_image.shape[1], np_image.shape[0]))

    prediction_foreground = np.zeros((resized_foreground_img.shape[0], resized_foreground_img.shape[1]), dtype=np.uint8)
    prediction_foreground[np.any(resized_foreground_img != [0, 0, 0, 0], axis=-1)] = 255
    print("prediction_foreground:", prediction_foreground)
    Image.fromarray(prediction_foreground).save(f"/usr/prakt/s0075/image-editing/code/outputs/foreground/blurtest-prediction-fg.png")

    prediction[prediction_foreground == 255] = 255
    print("prediction:", prediction)
    Image.fromarray(prediction).save(f"/usr/prakt/s0075/image-editing/code/outputs/foreground/blurtest-prediction2.png")
    """

    # blur image given depth map
    oimg = Image.fromarray(resized_image)
    mimg = Image.fromarray(prediction)

    bimg = oimg.filter(ImageFilter.BoxBlur(int(boxBlur)))
    bimg = bimg.filter(ImageFilter.BLUR)
    for _ in range(sharpen):
        bimg = bimg.filter(ImageFilter.SHARPEN)

    rimg = Image.composite(oimg, bimg, mimg)
    outimg = rimg.resize((image.shape[1], image.shape[0]))
    return outimg


# use the same 
def get_fgbg(input_image):
    foreground_img = extract_foreground(input_image)
    rescaled_img, white_bg_image = scale_and_paste(foreground_img, input_image)
    return rescaled_img, white_bg_image


input_img = Image.open("/usr/prakt/s0075/image-editing/code/inputs/portrait-examples/jake.jpg").convert("RGB")
bBlur = 15
sharpen = 0
foreground_img = extract_foreground(input_img)
foreground_img.save(f"/usr/prakt/s0075/image-editing/code/outputs/foreground/jake-fg.png")


blurred_img = apply_blur(input_img, foreground_img, bBlur, sharpen)

# paste the original foreground element on top of the blurred image
blurred_img.paste(foreground_img, foreground_img)
blurred_img.save(f"/usr/prakt/s0075/image-editing/code/outputs/foreground/jake-b{bBlur}-s{sharpen}.png")


