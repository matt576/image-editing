from diffusers import DiffusionPipeline
import torch
from PIL import Image
import numpy as np
from operations_image import expand_white_areas, expand_white_areas_outpainting
from extract_foreground import extract_foreground_mask, extract_foreground_image


def background_replace_mask_stablediffusion(input_image: Image, mask_image: Image, prompt: str, steps: int) -> Image:

    pipe = DiffusionPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16"
    )
    pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(0)
    size = np.array(input_image).shape[:2]
    
    # reverse the mask for outpainting
    reversed_mask_array = 255 - np.array(mask_image)
    reversed_mask_array = Image.fromarray(reversed_mask_array)
    reversed_mask_array = expand_white_areas_outpainting(reversed_mask_array, 2) #optional
    resized_input_image = input_image.resize((1024, 1024))
    resized_reversed_mask_array = reversed_mask_array.resize((1024, 1024))
    output_image = pipe(prompt=prompt, 
                        image=resized_input_image, 
                        mask_image=resized_reversed_mask_array,  
                        guidance_scale=7.5,
                        num_inference_steps=steps,
                        strength=0.99,
                        negative_prompt="blurry, poor quality, poor details, deformed, extra limbs",
                        generator=generator).images[0]
    resized_output_image = output_image.resize((size[1], size[0]))
    return resized_output_image


# method extracting foreground (RMBG-1.4) and outpainting the background (stable-diffusion-2-inpaint)
def background_replace_portrait_stablediffusion(input_image: Image, prompt:str, steps) -> Image:
    forground_image = extract_foreground_image(input_image)
    foreground_mask = extract_foreground_mask(forground_image)
    output_image = background_replace_mask_stablediffusion(input_image, foreground_mask, prompt, steps)
    return output_image

def background_replace_sdxl_gradio(input_image, prompt, steps):
    steps = int(steps)
    input_image = input_image.convert("RGB")

    output_image = background_replace_portrait_stablediffusion(input_image, prompt, steps)
    outdir = "outputs/gradio"
    filename = "background_replace_sdxl_output_gradio.png"
    output_image.save(f"{outdir}/{filename}")
    print("Output image saved to: ", outdir + "/" + filename)

    return output_image

if __name__ == "__main__":
    
    image = Image.open(f"inputs/inpainting/jessi.png")
    image = image.convert("RGB")
    prompt = "dog sitting on the middle of a busy city street in tokyo, sunny day, blue sky"
    output_image = background_replace_portrait_stablediffusion(image, prompt, 50)
    output_image.save("outputs/eval/jessi_background_replacement_sdxl.png")


    
