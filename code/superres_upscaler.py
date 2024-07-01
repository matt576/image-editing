import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

if __name__ == "__main__":

    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")

    low_res_img = Image.open("test_dataset/jessi.png").convert("RGB")
    # original_width, original_height = low_res_img.size
    # desired_width_scale = original_width / 128 ##new
    # desired_height_scale = original_height / 128 ##new
    low_res_img = low_res_img.resize((128, 128))
    low_res_img.save("outputs/superres/superres_upscaler_temp128.png")

    prompt = "a black dog"

    upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
    upscaled_image.save("outputs/superres/superres_upscaler_output.png")

    # final_width = int(upscaled_image.width * desired_width_scale)
    # final_height = int(upscaled_image.height * desired_height_scale)
    # upscaled_image_resized = upscaled_image.resize((final_width, final_height))

    # upscaled_image_resized.save("outputs/superres/superres_upscaler_output_resized.png")