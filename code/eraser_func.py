from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

init_image = Image.open("inputs/batman.jpg")
init_image = init_image.resize((512, 512))

mask_image = Image.open("inputs/batman_mask.jpg")
mask_image = mask_image.resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
)
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=None, image=init_image, mask_image=mask_image).images[0]
image.save("./yellow_cat_on_park_bench.png")
