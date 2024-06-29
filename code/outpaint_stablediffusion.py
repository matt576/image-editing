from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import numpy as np


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")


image = Image.open(f"inputs/example_dog/dog.png")
mask_image = Image.open(f"inputs/example_dog/dog_mask.png")
reversed_mask_array = 255 - np.array(mask_image)
reversed_mask_array = Image.fromarray(reversed_mask_array)

prompt = "dog sitting on a dirty street, cars in the background"
# image and mask_image should be PIL images.
# the mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=reversed_mask_array).images[0]
image.save("outputs/example_dog/outpaint-dog-street.png")
