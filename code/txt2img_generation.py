from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
generator = torch.Generator("cuda").manual_seed(31)
image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", 
                generator=generator,
                negative_prompt="ugly, poor quality",
                num_inference_steps=20,
                guidance_scale=3.5,
                height=512,
                width=512).images[0]
image.save("outputs/txt2img/test.png")