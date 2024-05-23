from diffusers import DiffusionPipeline
import torch

# load model and scheduler
ldm = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

# run pipeline in inference (sample random noispythone and denoise)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
ldm.to(device)


prompt = "titanic"
images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6).images

# save images
for idx, image in enumerate(images):
    image.save(f"my_scripts/samples/titanic2-{idx}.png")
