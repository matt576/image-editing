import torch
import numpy as np
from PIL import Image, ImageFilter
import os, sys, copy
import cv2
from extract_foreground import extract_foreground_image, scale_and_paste

# adapth the repository path
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'Depth-Anything')
sys.path.append(config_path)


from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

# change the current path to Depth-Anything to correctly maintain the model
os.chdir(config_path)
model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14")
os.chdir(current_dir)


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



def apply_blur(input_img: Image, foreground_img: Image, boxBlur: int, sharpen: int, set_focal_point: bool) -> Image:

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
    
    if set_focal_point:
        focus = copy.deepcopy(prediction)

        # apply the foreground mask to depth estimation
        foreground_img = foreground_img.resize((resized_image.shape[1], resized_image.shape[0]))
        np_im = np.array(foreground_img)
        foreground_mask = np.zeros((np_im.shape[0], np_im.shape[1]), dtype=np.uint8)
        foreground_mask[np_im[:, :, 3] > 0] = 1
        focus[foreground_mask == 0] = 0

        # calculate the average pixel value (distance) of the foreground element
        non_zero_pixels = focus[focus > 0]
        average_focus = np.mean(non_zero_pixels)

        # modify the depth map to focus on the foreground object
        prediction = np.clip(255 - np.abs(prediction - average_focus).astype("uint8"), 0, 255)

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
    foreground_img = extract_foreground_image(input_image)
    rescaled_img, white_bg_image = scale_and_paste(foreground_img, input_image)
    return rescaled_img, white_bg_image


def blur_image(input_image: Image, bBlur: int, sharpen: int = 0, set_focal_point=False) -> Image:
    
    # extract foreground from the image
    foreground_img = extract_foreground_image(input_image)

    # apply blur to the whole image
    blurred_img = apply_blur(input_image, foreground_img, bBlur, sharpen, set_focal_point)

    # paste the original foreground element on top of the blurred image
    blurred_img.paste(foreground_img, foreground_img)
    return blurred_img



if __name__ == "__main__":

    directory_path = "/usr/prakt/s0075/image-editing/code/inputs/background-blurring/"

    for filename in os.listdir(directory_path):
        print(filename)
        name = filename.split('.')[0]
        input_image = Image.open(f"{directory_path}/{filename}").convert("RGB")
        blurred_image = blur_image(input_image, 15, 0)
        blurred_image.save(f"/usr/prakt/s0075/image-editing/code/outputs/background-blurring/{name}.png")
