from transformers import pipeline
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import torch
import os
import numpy as np
import torch.nn.functional as F
from PIL import Image
import random
from diffusers import ControlNetModel
from scipy.ndimage import label, find_objects


def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), mode='bilinear', size=model_input_size)
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list)-> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)
    im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array


# extract foreground image (RMBG-1.4)
def extract_foreground_image(input_image: Image) -> Image:

    # set the values for model size and images of how much percent of whole image should be dismissed
    model_size = (1024, 1024)
    img_percent = 0.05

    if img_percent < 0.0 or img_percent > 1.0:
        print('The img_percent variable should be in the range (0.0, 1.0)')
        return None

    dismissed_pixels = img_percent * model_size[0] * model_size[1]

    # prepare input
    input_image_size = np.array(input_image).shape[:2]
    input_image = input_image.resize(model_size)

    orig_im = np.array(input_image)
    orig_im_size = orig_im.shape[0:2]
    model_input_size = model_size
    image = preprocess_image(orig_im, model_input_size).to(device)

    # inference 
    result = model(image)

    # post process
    result_image = postprocess_image(result[0][0], orig_im_size)

    # save result
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
    
    # remove small groups of foreground map
    np_im = np.array(pil_im)
    np_im[np_im > 0] = 1
    labeled_array, num_features = label(np_im)
    objects = find_objects(labeled_array)
    for i, obj_slice in enumerate(objects):
        if np.sum(labeled_array[obj_slice] == (i + 1)) < dismissed_pixels:      # The whole image has 1024x1024 ~ 10^6 pixels
            np_im[labeled_array == (i + 1)] = 0
    masked_im = np_im * np.array(pil_im)
    cleaned_pil_im = Image.fromarray(masked_im)
    
    # apply the mask
    no_bg_image.paste(input_image, mask=cleaned_pil_im)
    no_bg_image = no_bg_image.resize((input_image_size[1], input_image_size[0]))

    return no_bg_image


def extract_foreground_mask(input_image: Image) -> Image:

    input_image = input_image.convert('RGBA')
    img_array = np.array(input_image)
    alpha_channel = img_array[:, :, 3]
    mask_array = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)
    output_mask = Image.fromarray(mask_array, mode='L')
    return output_mask


def scale_and_paste(original_image, background_image=None):
    
    aspect_ratio = original_image.width / original_image.height

    if original_image.width > original_image.height:
        new_width = 512
        new_height = round(new_width / aspect_ratio)
    else:
        new_height = 512
        new_width = round(new_height * aspect_ratio)

    # make the subject a little smaller
    #new_width = new_width - 20
    #new_height = new_height - 20

    resized_original = original_image.resize((new_width, new_height), Image.LANCZOS)
    if background_image is None:
        white_background = Image.new("RGBA", (512, 512), "white")
    else:
        white_background = background_image.resize((512, 512))

    x = (512 - new_width) // 2
    y = (512 - new_height) // 2
    white_background.paste(resized_original, (x, y), resized_original)
    return resized_original, white_background


def generate_image(prompt, negative_prompt, inpaint_image, zoe_image, seed: int = None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        image=[inpaint_image, zoe_image],
        guidance_scale=6.5,
        num_inference_steps=25,
        generator=generator,
        controlnet_conditioning_scale=[0.5, 0.8],
        control_guidance_end=[0.9, 0.6],
    ).images[0]
    return image


controlnet = ControlNetModel.from_pretrained(
    "destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16"
)

model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4",trust_remote_code=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_fgbg(input_image):
    foreground = extract_foreground_image(input_image)
    rescaled_img, white_bg_image = scale_and_paste(foreground)
    return rescaled_img, white_bg_image


if __name__ == "__main__":
    import os
    directory_path = "/usr/prakt/s0075/image-editing/code/inputs/background-blurring/"

    for filename in os.listdir(directory_path):
        print(filename)
        input_image = Image.open(f"{directory_path}/{filename}").convert("RGB")
        extracted_img, _ = get_fgbg(input_image)
        extracted_img.save(f"/usr/prakt/s0075/image-editing/code/outputs/foreground/{filename}")
