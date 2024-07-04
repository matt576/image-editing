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

    init_image = Image.open("test_dataset/pelican.png")
    init_image = init_image.convert("RGB")

    original_width, original_height = init_image.size

    if original_width > original_height:
        init_image = init_image.resize((768, 512))
    elif original_width < original_height:
        init_image = init_image.resize((512, 768))
    else:
        init_image = init_image.resize((512, 512))

    prompt = "Photorealistic pelican on a pier by the beach, rain pouring down, dark clouds with streaks of lightning."
    strength = 0.75
    gs= 7.5
    np = "poor details, blurry"
    stp = 100

    images = pipeline(prompt=prompt, image=init_image, strength=strength, guidance_scale=gs, num_inference_steps=stp, negative_prompt=np).images
    output_image = images[0].resize((original_width, original_height))

    output_dir = "outputs/restyling"
    filename = "restyling_output_sdxl_pelican.png"
    output_image.save(f"{output_dir}/{filename}")

def restyling_sdxl_gradio(input_image, text_prompt, strength, guidance_scale, negative_prompt, steps):
    pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipeline.enable_model_cpu_offload()

    strength = float(strength)
    guidance_scale = float(guidance_scale)
    steps = int(steps)
    input_image = input_image.convert("RGB")
    original_width, original_height = input_image.size

    if original_width > original_height:
        input_image = input_image.resize((768, 512))
    elif original_width < original_height:
        input_image = input_image.resize((512, 768))
    else:
        input_image = input_image.resize((512, 512))
    
    images = pipeline(prompt=text_prompt, image=input_image, strength=strength, guidance_scale=guidance_scale, negative_prompt = negative_prompt, num_inference_steps=steps).images

    output_image = images[0].resize((original_width, original_height))

    output_dir = "outputs/gradio"
    filename = "restyling_output_sdxl_new.png"
    output_image.save(f"{output_dir}/{filename}")

    return output_image