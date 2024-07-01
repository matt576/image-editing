import random

import requests
import torch
from controlnet_aux import ZoeDetector
from PIL import Image, ImageOps

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
)

def scale_and_paste(original_image, size):
    aspect_ratio = original_image.width / original_image.height

    if original_image.width > original_image.height:
        new_width = size[0]
        new_height = round(new_width / aspect_ratio)
    else:
        new_height = size[1]
        new_width = round(new_height * aspect_ratio)

    resized_original = original_image.resize((new_width, new_height), Image.LANCZOS)
    white_background = Image.new("RGBA", (64, 64), "white")
    x = (size[0] - new_width) // 2
    y = (size[1] - new_height) // 2
    white_background.paste(resized_original, (x, y), resized_original)

    return resized_original, white_background



original_image = Image.open(
    requests.get(
        "https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/no-background-jordan.png",
        stream=True,
    ).raw
).convert("RGBA")


size = (64, 64)
resized_img, white_bg_image = scale_and_paste(original_image, size)

zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
image_zoe = zoe(white_bg_image, detect_resolution=64, image_resolution=64)
image_zoe.save('outputs/test/image_zoe.png')


controlnets = [
    ControlNetModel.from_pretrained(
        "destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16"
    ),
    ControlNetModel.from_pretrained(
        "diffusers/controlnet-zoe-depth-sdxl-1.0", torch_dtype=torch.float16
    ),
]
# vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(

    "SG161222/Realistic_Vision_V5.1_noVAE", torch_dtype=torch.float16, controlnet=controlnets#, vae=vae
).to("cuda")


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

prompt = "nike air jordans on a basketball court"
negative_prompt = ""

temp_image = generate_image(prompt, negative_prompt, white_bg_image, image_zoe, 908097)


x = (size[0] - resized_img.width) // 2
y = (size[1] - resized_img.height) // 2
temp_image.paste(resized_img, (x, y), resized_img)
temp_image.save('outputs/test/temp_image.png')
