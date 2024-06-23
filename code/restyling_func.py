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
    init_image = init_image.resize((768, 512))

    prompt = "Photorealistic Gotham City night skyline, rain pouring down, dark clouds with streaks of lightning."

    images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
    images[0].save("restyling_output.png")

def restyling_gradio(input_image, text_prompt):
    device = "cuda"
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    input_image = input_image.resize((768, 512))

    images = pipe(prompt=text_prompt, image=input_image, strength=0.75, guidance_scale=7.5).images

    output_dir = "outputs/gradio"
    filename = "restyling_output.png"
    images[0].save(f"{output_dir}/{filename}")

    return images[0]