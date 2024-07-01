import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import AutoPipelineForImage2Image

if __name__ == "__main__":
    pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipeline.enable_model_cpu_offload()

    init_image = Image.open("inputs/city.png")

    original_width, original_height = init_image.size

    if original_width > original_height:
        init_image = init_image.resize((768, 512))
    elif original_width < original_height:
        init_image = init_image.resize((512, 768))
    else:
        init_image = init_image.resize((512, 512))

    prompt = "Photorealistic Gotham City night skyline, rain pouring down, dark clouds with streaks of lightning."

    images = pipeline(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
    output_image = images[0].resize((original_width, original_height))

    output_dir = "outputs/restyling"
    filename = "restyling_output_sdxl.png"
    output_image.save(f"{output_dir}/{filename}")

def restyling_sdxl_gradio(input_image, text_prompt):
    pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipeline.enable_model_cpu_offload()

    original_width, original_height = input_image.size

    if original_width > original_height:
        input_image = input_image.resize((768, 512))
    elif original_width < original_height:
        input_image = input_image.resize((512, 768))
    else:
        input_image = input_image.resize((512, 512))
    
    images = pipeline(prompt=text_prompt, image=input_image, strength=0.75, guidance_scale=7.5).images

    output_image = images[0].resize((original_width, original_height))

    output_dir = "outputs/gradio"
    filename = "restyling_output_sdxl.png"
    output_image.save(f"{output_dir}/{filename}")

    return output_image