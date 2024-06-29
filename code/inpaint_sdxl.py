import os, sys
import torch
from diffusers import AutoPipelineForInpainting
# from diffusers.utils import load_image, make_image_grid
# import numpy as np
from PIL import Image


def inpaint_sdxl_gradio(input_image, mask_image, text_input):
    output_dir = "outputs/gradio/inpainting"
    filename = "inpainted_sdxl_gradio.png"
    prompt = text_input
    input_image = input_image.convert("RGB")
    original_width, original_height = input_image.size
    input_image = input_image.resize((512, 512))

    mask_image = mask_image.convert("L")
    mask_image = mask_image.resize(input_image.size)

    pipeline = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
    )
    pipeline.enable_model_cpu_offload()

    generator = torch.Generator("cuda").manual_seed(92)

    # mask_image = pipeline.mask_processor.blur(mask_image, blur_factor=33) #Optional
    print(prompt)
    image = pipeline(prompt=prompt, image=input_image, mask_image=mask_image, generator=generator).images[0]
    
    image_resized = image.resize((original_width, original_height))
    image_resized.save(f"{output_dir}/{filename}")
    return image_resized


if __name__ == "__main__":

    output_dir = "outputs/controlnet"
    filename = "jessi_inpainted_sdxl_dilated.png"
    filename_resized = "jessi_inpainted_sdxl_resized.png"
    prompt = "cat in color orange, photorealistic, detailed, high quality"

    init_image = Image.open("test_dataset/jessi.png")
    init_image = init_image.convert("RGB")
    original_width, original_height = init_image.size

    init_image = init_image.resize((512, 512))

    mask_image = Image.open("inputs/eval/jessi_mask.png")
    mask_image = mask_image.convert("L")
    # mask_original_width, mask_original_height = mask_image.size
    mask_image = mask_image.resize(init_image.size)

    pipeline = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
    )
    pipeline.enable_model_cpu_offload()

    generator = torch.Generator("cuda").manual_seed(92)

    # mask_image = pipeline.mask_processor.blur(mask_image, blur_factor=33) #Optional

    image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
    image.save(f"{output_dir}/{filename}")
    image_resized = image.resize((original_width, original_height))
    image_resized.save(f"{output_dir}/{filename_resized}")