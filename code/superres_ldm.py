import requests
from PIL import Image
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline
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

def superres_gradio(input_image, steps):
    steps = int(steps)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/ldm-super-resolution-4x-openimages"
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    low_res_img = input_image.convert("RGB")
    original_width, original_height = low_res_img.size

    closest_width = int(round(original_width / 128)) * 128 ###
    closest_height = int(round(original_height / 128)) * 128 ###
    low_res_img = low_res_img.resize((closest_width, closest_height))###
    new_size = (closest_width, closest_height) ###

    patch_size = 128
    patches = divide_image(low_res_img, patch_size)

    upscaled_patches = []
    for patch in patches:
        upscaled_patch = pipeline(image=patch, num_inference_steps=steps, eta=1).images[0]
        upscaled_patches.append(upscaled_patch)

    upscaled_image = reassemble_image(upscaled_patches, new_size[0], new_size[1], patch_size)
    upscaled_image = upscaled_image.resize((original_width * 4, original_height * 4))

    os.makedirs("outputs/gradio/superres", exist_ok=True)
    upscaled_image.save("outputs/gradio/superres/superres_ldm_output_gradio.png")
    return upscaled_image

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/ldm-super-resolution-4x-openimages"
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    low_res_img = Image.open("inputs/superresolution/lenna.png").convert("RGB")
    original_width, original_height = low_res_img.size

    closest_width = int(round(original_width / 128)) * 128 ###
    closest_height = int(round(original_height / 128)) * 128 ###
    low_res_img = low_res_img.resize((closest_width, closest_height))###
    new_size = (closest_width, closest_height) ###

    # if original_width > original_height:
    #     low_res_img = low_res_img.resize((768, 512))
    #     new_size = (768, 512)
    # elif original_width < original_height:
    #     low_res_img = low_res_img.resize((512, 768))
    #     new_size = (512, 768)
    # else:
    #     low_res_img = low_res_img.resize((512, 512))
    #     new_size = (512, 512)
    
    patch_size = 128
    patches = divide_image(low_res_img, patch_size)

    num_inference_steps = 100
    # counter = 0
    upscaled_patches = []
    for patch in patches:
        # patch.save(f"patch{counter}.png")
        # counter = counter + 1
        upscaled_patch = pipeline(image=patch, num_inference_steps=num_inference_steps, eta=1).images[0]
        upscaled_patches.append(upscaled_patch)

    upscaled_image = reassemble_image(upscaled_patches, new_size[0], new_size[1], patch_size)
    upscaled_image = upscaled_image.resize((original_width * 4, original_height * 4))

    os.makedirs("outputs/superres", exist_ok=True)
    upscaled_image.save("outputs/superres/superres_ldm_output.png")


"""
Old approach:

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/ldm-super-resolution-4x-openimages"

    # load model and scheduler
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    # let's download an  image
    # url = "https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"
    # response = requests.get(url)
    # low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = Image.open("test_dataset/jessi.png").convert("RGB")
    original_width, original_height = low_res_img.size
    desired_width_scale = original_width / 128 ##new
    desired_height_scale = original_height / 128 ##new
    low_res_img = low_res_img.resize((128, 128))
    # low_res_img.save("inputs/superres/ldm_low_res.png")

    # run pipeline in inference (sample random noise and denoise)
    upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
    final_width = int(upscaled_image.width * desired_width_scale)
    final_height = int(upscaled_image.height * desired_height_scale)
    upscaled_image = upscaled_image.resize((final_width, final_height))
    # save image

    upscaled_image.save("outputs/superres/ldm_generated_image.png")

def superres_gradio(input_image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/ldm-super-resolution-4x-openimages"

    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    low_res_img = input_image
    original_width, original_height = low_res_img.size
    desired_width_scale = original_width / 128 ##new
    desired_height_scale = original_height / 128 ##new
    low_res_img = low_res_img.resize((128, 128))

    upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
    final_width = int(upscaled_image.width * desired_width_scale)
    final_height = int(upscaled_image.height * desired_height_scale)
    upscaled_image = upscaled_image.resize((final_width, final_height))

    output_dir = "outputs/gradio"
    filename = "superres_output.png"
    upscaled_image.save(f"{output_dir}/{filename}")

    return upscaled_image

    """