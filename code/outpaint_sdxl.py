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
    #white_img = Image.new('RGB', extended_image.size, color='white')
    #extended_image.paste(white_img)
    # extended_image = extended_image.filter(ImageFilter.BoxBlur(100))
    extended_image.paste(input_image, (coordinates[0], coordinates[2]))
    extended_image.save("outputs/outpainting/TEST_WHITE.png")

    # new mask image
    extended_mask = Image.new('L', (new_width, new_height), color='white')
    input_mask = Image.new('L', input_image.size, color='black')
    extended_mask.paste(input_mask, (coordinates[0], coordinates[2]))
    extended_mask = expand_white_areas(extended_mask, 15)

    # extended_image = extended_image.resize((512, 512))
    # extended_mask = extended_mask.resize((512, 512))

    print(extended_mask.size, extended_mask.info)
    print(extended_image.size, extended_image.info)

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

    
    image = Image.open(f"inputs/outpainting/taxi.png")
    prompt = "street with two people walking"
    extend_right, extend_left, extend_up, extend_down = 0, 300, 0, 0
    coordinates = [extend_right, extend_left, extend_up, extend_down]
    output_image = outpaint_stablediffusionxl(image, prompt, coordinates, 15)
    output_image.save("outputs/outpainting/sdxl-taxi-test.png")