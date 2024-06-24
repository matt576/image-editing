from transformers import pipeline
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import random
from controlnet_aux import ZoeDetector
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL


def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    # orig_im_size=im.shape[0:2]
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

def extract_foreground(input_path: str):

    # prepare input
    orig_im = Image.open(input_path).convert("RGB")
    orig_im = np.array(orig_im)

    orig_im_size = orig_im.shape[0:2]
    # model_input_size = orig_im_size     # decrease the model size in case of lack of memory
    model_input_size = (1024, 1024)
    image = preprocess_image(orig_im, model_input_size).to(device)

    # inference 
    result = model(image)

    # post process
    result_image = postprocess_image(result[0][0], orig_im_size)

    # save result
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
    orig_image = Image.open(input_path).convert("RGB")
    no_bg_image.paste(orig_image, mask=pil_im)
    # no_bg_image.save(output_path)
    return no_bg_image

def scale_and_paste(original_image):
    aspect_ratio = original_image.width / original_image.height

    if original_image.width > original_image.height:
        new_width = 512
        new_height = round(new_width / aspect_ratio)
    else:
        new_height = 512
        new_width = round(new_height * aspect_ratio)

    # make the subject a little smaller
    new_width = new_width - 20
    new_height = new_height - 20

    resized_original = original_image.resize((new_width, new_height), Image.LANCZOS)
    white_background = Image.new("RGBA", (512, 512), "white")
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



input_dir = "/usr/prakt/s0075/image-editing/code/inputs/foreground/safety_car.jpg"
output_dir = "/usr/prakt/s0075/image-editing/code/outputs/foreground/safety_car-foreground.png"

controlnet = ControlNetModel.from_pretrained(
    "destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16"
)

model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4",trust_remote_code=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_foreground(input_dir):
    foreground = extract_foreground(input_dir)
    rescaled_img, white_bg_image = scale_and_paste(foreground)
    return rescaled_img, white_bg_image


# rescaled_img, white_bg_image = get_foreground(input_dir)
# white_bg_image.save("/usr/prakt/s0075/image-editing/code/outputs/foreground/safety_car1.png")
# rescaled_img.save("/usr/prakt/s0075/image-editing/code/outputs/foreground/safety_car0.png")
