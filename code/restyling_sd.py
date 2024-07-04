import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

if __name__ == "__main__":
    device = "cuda"
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    # url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    # response = requests.get(url)
    # init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = Image.open("test_dataset/pelican.png")
    init_image = init_image.convert("RGB")
    original_width, original_height = init_image.size

    if original_width > original_height:
        init_image = init_image.resize((768, 512))
    elif original_width < original_height:
        init_image = init_image.resize((512, 768))
    else:
        init_image = init_image.resize((512, 512))

    prompt = "pelican in a pier, impressionistic style"
    strength = 0.75
    gs= 7.5
    np = "poor details, blurry"
    stp = 100

    images = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=gs, num_inference_steps=stp, negative_prompt=np).images
    output_image = images[0].resize((original_width, original_height))

    output_dir = "outputs/restyling"
    filename = "restyling_output.png"
    output_image.save(f"{output_dir}/{filename}")

def restyling_gradio(input_image, text_prompt, strength, guidance_scale, negative_prompt, steps):
    device = "cuda"
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)

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
    
    images = pipe(prompt=text_prompt, image=input_image, strength=strength, guidance_scale=guidance_scale, negative_prompt = negative_prompt, num_inference_steps=steps).images

    output_image = images[0].resize((original_width, original_height))

    output_dir = "outputs/gradio"
    filename = "restyling_output_sdv15_new.png"
    output_image.save(f"{output_dir}/{filename}")

    return output_image