import random
import numpy as np
import torch
from controlnet_aux import ZoeDetector
from PIL import Image, ImageOps

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
)

print(torch.cuda.is_available())


# load controlnets
controlnets = [
    ControlNetModel.from_pretrained(
        "destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16"
    ),
    ControlNetModel.from_pretrained("diffusers/controlnet-zoe-depth-sdxl-1.0", torch_dtype=torch.float16),
]

# vae in case it doesn't come with model
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")


def scale_and_paste(original_image):
    aspect_ratio = original_image.width / original_image.height

    if original_image.width > original_image.height:
        new_width = 256
        new_height = round(new_width / aspect_ratio)
    else:
        new_height = 256
        new_width = round(new_height * aspect_ratio)

    # make the subject a little smaller
    new_width = new_width - 20
    new_height = new_height - 20

    resized_original = original_image.resize((new_width, new_height), Image.LANCZOS)
    white_background = Image.new("RGBA", (256, 256), "white")
    x = (256 - new_width) // 2
    y = (256 - new_height) // 2
    white_background.paste(resized_original, (x, y), resized_original)
    return resized_original, white_background


# function to generate
def generate_image(prompt, negative_prompt, inpaint_image, zoe_image, seed: int = None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        image=[inpaint_image, zoe_image],
        guidance_scale=6.5,
        num_inference_steps=5,
        generator=generator,
        controlnet_conditioning_scale=[0.5, 0.8],
        control_guidance_end=[0.9, 0.6],
    ).images[0]
    return image

# function for final outpainting
def generate_outpaint(prompt, negative_prompt, image, mask, seed: int = None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipeline_outpaint(
        prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        guidance_scale=10.0,
        strength=0.8,
        num_inference_steps=5,
        generator=generator,
    ).images[0]
    return image

print("1")
input_dir = "/usr/prakt/s0075/image-editing/code/inputs/foreground/safety_car.jpg"
output_dir = "/usr/prakt/s0075/image-editing/code/inputs/foreground/safety_car-result.png"

# load the original image with alpha
orig_im = Image.open(input_dir).convert("RGBA")
resized_img, white_bg_image = scale_and_paste(orig_im)

# load preprocessor and generate depth map
image_zoe = zoe(white_bg_image, detect_resolution=128, image_resolution=256)   # TODO:takes long

print("2")

# initial pipeline for temp background
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", torch_dtype=torch.float16, variant="fp16", controlnet=controlnets, vae=vae
).to("cuda")

print("3")
# clear old pipeline for VRAM savings
torch.cuda.empty_cache()

# initial prompt
prompt = "a car on the highway"
negative_prompt = ""
temp_image = generate_image(prompt, negative_prompt, white_bg_image, image_zoe)

print("4")

# paste original subject over temporal background
x = (256 - resized_img.width) // 2
y = (256 - resized_img.height) // 2
temp_image.paste(resized_img, (x, y), resized_img)

# create a mask for the final outpainting
mask = Image.new("L", temp_image.size)
mask.paste(resized_img.split()[3], (x, y))
mask = ImageOps.invert(mask)
final_mask = mask.point(lambda p: p > 128 and 255)

print("5")

# clear old pipeline for VRAM savings
pipeline = None
torch.cuda.empty_cache()

# new pipeline with inpaiting model
pipeline_outpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
    "OzzyGT/RealVisXL_V4.0_inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
    vae=vae,
).to("cuda")

print("6")

# Use a blurred mask for better blend
mask_blurred = pipeline.mask_processor.blur(final_mask, blur_factor=20)

# better prompt for final outpainting
prompt = "high quality photo of a car on the highway, shadows, highly detailed"
negative_prompt = ""

print("7")

# generate the image
final_image = generate_outpaint(prompt, negative_prompt, temp_image, mask_blurred)

print("8")

# paste original subject over final background
x = (256 - resized_img.width) // 2
y = (256 - resized_img.height) // 2
final_image.paste(resized_img, (x, y), resized_img)
final_image.save(output_dir)
