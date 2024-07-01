from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image, ImageFilter
import numpy as np
from operations_image import expand_white_areas
from diffusers import AutoPipelineForInpainting



pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")


def outpaint_stablediffusionxl(input_image: Image, prompt: str, coordinates: list, steps: int) -> Image:

    # new width and height of the image
    width, height = input_image.size
    new_width = width + coordinates[0] + coordinates[1]
    new_height = height + coordinates[2] + coordinates[3]

    # new image with extended blank spaces
    extended_image = input_image.resize((new_width, new_height))
    # extended_image = extended_image.filter(ImageFilter.BoxBlur(10))
    extended_image.paste(input_image, (coordinates[0], coordinates[2]))

    # new mask image
    extended_mask = Image.new('L', (new_width, new_height), color='white')
    input_mask = Image.new('L', input_image.size, color='black')
    extended_mask.paste(input_mask, (coordinates[0], coordinates[2]))
    extended_mask = expand_white_areas(extended_mask, 5)

    # extended_image = extended_image.resize((512, 512))
    # extended_mask = extended_mask.resize((512, 512))

    generator = torch.Generator(device="cuda").manual_seed(0)
    output_image = pipe(
        prompt=prompt,
        image=extended_image,
        mask_image=extended_mask,
        guidance_scale=8.0,
        num_inference_steps=steps,
        strength=0.99,
        # generator=generator
    ).images[0]

    output_image = output_image.resize((new_width, new_height))
    return output_image


if __name__ == "__main__":

    
    image = Image.open(f"inputs/outpainting/dog.png")
    prompt = "forest"
    extend_right, extend_left, extend_up, extend_down = 0, 200, 0, 0
    coordinates = [extend_right, extend_left, extend_up, extend_down]
    output_image = outpaint_stablediffusionxl(image, prompt, coordinates, 15)
    output_image.save("outputs/outpainting/temporary-final.png")