import os, sys
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image

"""
Using the inpainting specific finetuded stable diffusion v1.5 model from
https://github.com/runwayml/stable-diffusion?tab=readme-ov-file
"""

def inpaint_sd_gradio(input_image, mask_image, text_input):
    output_dir = "outputs/gradio/inpainting"
    filename = "inpainted_sd_gradio.png"
    prompt = text_input
    input_image = input_image.convert("RGB")
    original_width, original_height = input_image.size
    input_image = input_image.resize((512, 512))

    mask_image = mask_image.convert("L")
    mask_image = mask_image.resize(input_image.size)

    pipeline = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        variant="fp16",
        torch_dtype=torch.float16,
    )
    pipeline.enable_model_cpu_offload()
    print(prompt)
    image = pipeline(prompt=prompt, image=input_image, mask_image=mask_image).images[0]
    
    image_resized = image.resize((original_width, original_height))
    image_resized.save(f"{output_dir}/{filename}")
    return image_resized

if __name__ == "__main__":

    output_dir = "outputs/controlnet"
    filename = "jessi_inpainted_sd.png"
    filename_resized = "jessi_inpainted_sd_resized.png"
    prompt = "pikachu, photorealistic, detailed, high quality"

    init_image = Image.open("test_dataset/jessi.png")
    init_image = init_image.convert("RGB")
    original_width, original_height = init_image.size

    init_image = init_image.resize((512, 512))

    mask_image = Image.open("inputs/eval/jessi_mask.png")
    mask_image = mask_image.convert("L")
    # mask_original_width, mask_original_height = mask_image.size
    mask_image = mask_image.resize(init_image.size)

    pipeline = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        variant="fp16",
        torch_dtype=torch.float16,
    )
    pipeline.enable_model_cpu_offload()

    # mask_image = pipeline.mask_processor.blur(mask_image, blur_factor=20) #Optional

    image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
    image.save(f"{output_dir}/{filename}")
    image_resized = image.resize((original_width, original_height))
    image_resized.save(f"{output_dir}/{filename_resized}")