from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image, ImageFilter
import numpy as np
from operations_image import expand_white_areas


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")


def outpaint_stablediffusion(input_image: Image, prompt: str, coordinates: list, steps: int) -> Image:

    width, height = input_image.size
    new_width = width + coordinates[0] + coordinates[1]
    new_height = height + coordinates[2] + coordinates[3]

    # new image with extended blank spaces
    extended_image = Image.new('RGB', (new_width, new_height), color='white')
    random_noise_array = np.random.randint(0, 256, (new_height, new_width, 3), dtype=np.uint8)
    extended_image = Image.fromarray(random_noise_array)
    # extended_image = input_image.resize((new_width, new_height))
    # extended_image = extended_image.filter(ImageFilter.BoxBlur(10))
    extended_image.paste(input_image, (coordinates[0], coordinates[2]))
    extended_image.save("outputs/outpainting/extended_image.png")

    # new mask image
    extended_mask = Image.new('L', (new_width, new_height), color='white')
    extended_mask.paste(Image.new('L', input_image.size, color='black'), (coordinates[0], coordinates[2]))
    extended_mask = expand_white_areas(extended_mask, 5)
    extended_mask.save("outputs/outpainting/extended_mask.png")

    # extended_image = extended_image.resize((512, 512))
    # extended_mask = extended_mask.resize((512, 512))

    output_image = pipe(prompt=prompt, image=extended_image, mask_image=extended_mask,  guidance_scale=7.5, num_inference_steps=steps).images[0]
    resized_output_image = output_image.resize((new_width, new_height))
    return resized_output_image


if __name__ == "__main__":

    image = Image.open(f"inputs/outpainting/armchair.png")
    prompt = "room with an armchair and green plants in a pot"
    extend_left, extend_right, extend_up, extend_down = 0, 200, 0, 0
    coordinates = [extend_left, extend_right, extend_up, extend_down]
    output_image = outpaint_stablediffusion(image, prompt, coordinates, 10)
    output_image.save("outputs/outpainting/armchair-plants-sd.png")
