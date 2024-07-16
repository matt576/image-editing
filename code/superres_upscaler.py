import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch
import os, sys

def divide_image(image, patch_size):
    patches = []
    img_width, img_height = image.size
    for i in range(0, img_height, patch_size):
        for j in range(0, img_width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)
    return patches

def reassemble_image(patches, img_width, img_height, patch_size):
    num_patches_x = img_width // patch_size
    num_patches_y = img_height // patch_size
    upscaled_img = Image.new('RGB', (img_width * 4, img_height * 4))

    k = 0
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            patch = patches[k]
            upscaled_img.paste(patch, (j * patch_size * 4, i * patch_size * 4))
            k += 1
    return upscaled_img

if __name__ == "__main__":

    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")

    low_res_img = Image.open("test_dataset/jessi.png").convert("RGB")
    original_width, original_height = low_res_img.size

    if original_width > original_height:
        low_res_img = low_res_img.resize((768, 512))
        new_size = (768, 512)
    elif original_width < original_height:
        low_res_img = low_res_img.resize((512, 768))
        new_size = (512, 768)
    else:
        low_res_img = low_res_img.resize((512, 512))
        new_size = (512, 512)
    
    patch_size = 128
    patches = divide_image(low_res_img, patch_size)

    prompt = "enhance detail"
    num_inference_steps = 50
    upscaled_patches = []
    for patch in patches:
        upscaled_patch = pipeline(prompt=prompt, image=patch, num_inference_steps=num_inference_steps).images[0]
        upscaled_patches.append(upscaled_patch)

    upscaled_image = reassemble_image(upscaled_patches, new_size[0], new_size[1], patch_size)
    upscaled_image = upscaled_image.resize((original_width * 4, original_height * 4))

    os.makedirs("outputs/superres", exist_ok=True)
    upscaled_image.save("outputs/superres/superres_upscaler_output.png")

def superres_upscaler_gradio(input_image, steps):
    steps = int(steps)

    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")

    low_res_img = input_image.convert("RGB")
    original_width, original_height = low_res_img.size

    closest_width = int(round(original_width / 128)) * 128 ###
    closest_height = int(round(original_height / 128)) * 128 ###
    low_res_img = low_res_img.resize((closest_width, closest_height))###
    new_size = (closest_width, closest_height) ###

    # if original_width < 350 or original_height < 350:
    #     low_res_img = low_res_img.resize((256, 256))
    #     new_size = (256, 256)
    # else:
    #     if original_width > original_height:
    #         low_res_img = low_res_img.resize((768, 512))
    #         new_size = (768, 512)
    #     elif original_width < original_height:
    #         low_res_img = low_res_img.resize((512, 768))
    #         new_size = (512, 768)
    #     else:
    #         low_res_img = low_res_img.resize((512, 512))
    #         new_size = (512, 512)
    
    patch_size = 128
    patches = divide_image(low_res_img, patch_size)

    prompt = "enhance detail"
    upscaled_patches = []
    for patch in patches:
        upscaled_patch = pipeline(prompt=prompt, image=patch, num_inference_steps=steps).images[0]
        upscaled_patches.append(upscaled_patch)

    upscaled_image = reassemble_image(upscaled_patches, new_size[0], new_size[1], patch_size)
    upscaled_image = upscaled_image.resize((original_width * 4, original_height * 4))

    os.makedirs("outputs/gradio/superres", exist_ok=True)
    upscaled_image.save("outputs/gradio/superres/superres_upscaler_output_gradio.png")
    return upscaled_image