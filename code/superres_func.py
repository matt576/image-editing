import requests
from PIL import Image
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/ldm-super-resolution-4x-openimages"

# load model and scheduler
pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
pipeline = pipeline.to(device)

# let's download an  image
# url = "https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"
# response = requests.get(url)
# low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
low_res_img = Image.open("inputs/batman.jpg")
original_width, original_height = low_res_img.size
desired_width_scale = original_width / 128 ##new
desired_height_scale = original_height / 128 ##new
low_res_img = low_res_img.resize((128, 128))
low_res_img.save("ldm_low_res.png")

# run pipeline in inference (sample random noise and denoise)
upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
final_width = int(upscaled_image.width * desired_width_scale)
final_height = int(upscaled_image.height * desired_height_scale)
upscaled_image = upscaled_image.resize((final_width, final_height))
# save image

upscaled_image.save("ldm_generated_image.png")

def superres_gradio(input_image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/ldm-super-resolution-4x-openimages"

    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    low_res_img = input_image
    original_width, original_height = low_res_img.size
    desired_width_scale = original_width / 128 ##new
    desired_height_scale = original_height / 128 ##new
    low_res_img = low_res_img.resize((128, 128))

    upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
    final_width = int(upscaled_image.width * desired_width_scale)
    final_height = int(upscaled_image.height * desired_height_scale)
    upscaled_image = upscaled_image.resize((final_width, final_height))

    output_dir = "outputs/gradio"
    filename = "superres_output.png"
    upscaled_image.save(f"{output_dir}/{filename}")

    return upscaled_image
