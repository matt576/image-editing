# Load initial image and mask
import os, sys

import torchvision.transforms.v2.functional

# parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# sys.path.append(parent_dir)

from diffusers.utils import load_image, make_image_grid
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
import numpy as np
import torch

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float16) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float16) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1]
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    # print(f"Control image shape: {image.shape}")
    return image

def inputation(input_image, mask_image, text_prompt, pipe):
    control_image = make_inpaint_condition(input_image, mask_image)      # image with extracted mask from input
    # test = torchvision.transforms.v2.functional.to_pil_image(control_image[0])
    # test.save(f"{output_dir}/test.png")

    output = pipe(
        prompt=text_prompt,
        num_inference_steps=5,
        eta=1.0,
        image=input_image,
        mask_image=mask_image,
        control_image=control_image,
    ).images[0]
    return output

def controlnet_inpaint_gradio(input_image, mask_image, text_input):
    
    original_width, original_height = input_image.size
    if original_width > original_height:
        input_image = input_image.resize((768, 512))
    elif original_width < original_height:
        input_image = input_image.resize((512, 768))
    else:
        input_image = input_image.resize((512, 512))
    
    mask_original_width, mask_original_height = mask_image.size
    if mask_original_width > mask_original_height:
        mask_image = mask_image.resize((768, 512))
    elif mask_original_width < mask_original_height:
        mask_image = mask_image.resize((512, 768))
    else:
        mask_image = mask_image.resize((512, 512))

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint",
                                            torch_dtype=torch.float16,
                                            use_safetensors=True)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                                    controlnet=controlnet,
                                                                    torch_dtype=torch.float16,
                                                                    use_safetensors=True,
                                                                    safety_checker=None,
                                                                    requires_safety_checker=False)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    output = inputation(input_image, mask_image, text_input, pipe)
    output = output.resize((original_width, original_height))
    output_dir = "outputs/gradio"
    filename = "sdv15controlnet_inpaint_output.png"
    output.save(f"{output_dir}/{filename}")
    return output

if __name__ == "__main__":
    output_dir = "outputs/controlnet"
    filename = "jessi_inpainted_controlnet.png"
    text_prompt = "black bear cub, photorealistic, detailed, high quality"

    init_image = Image.open("test_dataset/jessi.png")
    init_image = init_image.convert("RGB")
    original_width, original_height = init_image.size
    # if original_width > original_height:
    #     init_image = init_image.resize((768, 512))
    # elif original_width < original_height:
    #     init_image = init_image.resize((512, 768))
    # else:
    init_image = init_image.resize((512, 512))

    mask_image = Image.open("outputs/grounded_sam/gradio/groundedsam_mask_resized.png")
    mask_image = mask_image.convert("L")
    mask_original_width, mask_original_height = mask_image.size
    # if mask_original_width > mask_original_height:
    #     mask_image = mask_image.resize((768, 512))
    # elif mask_original_width < mask_original_height:
    #     mask_image = mask_image.resize((512, 768))
    # else:
    #     mask_image = mask_image.resize((512, 512))

    mask_image = mask_image.resize(init_image.size)
    print(init_image.size, mask_image.size)

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint",
                                             torch_dtype=torch.float16,
                                             use_safetensors=True)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                                    controlnet=controlnet,
                                                                    torch_dtype=torch.float16,
                                                                    use_safetensors=True,
                                                                    safety_checker=None,
                                                                    requires_safety_checker=False)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    output = inputation(init_image, mask_image, text_prompt, pipe)
    output = output.resize((original_width, original_height))
    output.save(f"{output_dir}/{filename}") 
