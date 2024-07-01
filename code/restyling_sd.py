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
    init_image = Image.open("inputs/city.png")

    original_width, original_height = init_image.size

    if original_width > original_height:
        init_image = init_image.resize((768, 512))
    elif original_width < original_height:
        init_image = init_image.resize((512, 768))
    else:
        init_image = init_image.resize((512, 512))

    prompt = "Photorealistic Gotham City night skyline, rain pouring down, dark clouds with streaks of lightning."

    images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
    output_image = images[0].resize((original_width, original_height))

    output_dir = "outputs/restyling"
    filename = "restyling_output.png"
    output_image.save(f"{output_dir}/{filename}")

def restyling_gradio(input_image, text_prompt):
    device = "cuda"
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    original_width, original_height = input_image.size

    if original_width > original_height:
        input_image = input_image.resize((768, 512))
    elif original_width < original_height:
        input_image = input_image.resize((512, 768))
    else:
        input_image = input_image.resize((512, 512))
    
    images = pipe(prompt=text_prompt, image=input_image, strength=0.75, guidance_scale=7.5).images

    output_image = images[0].resize((original_width, original_height))

    output_dir = "outputs/gradio"
    filename = "restyling_output_sdv15.png"
    output_image.save(f"{output_dir}/{filename}")

    return output_image