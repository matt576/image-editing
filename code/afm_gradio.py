import gradio as gr
import torch
import os, sys
import requests
import numpy as np
from PIL import Image
import torchvision.transforms.v2.functional
from transformers import SamModel, SamProcessor
from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler

# imports from code scripts
from mask_func import get_mask
from inpaint_func import inputation #, make_inpaint_condition, controlnet, pipe


def run_afm_app(task_selector, input_image, mask_image, text_input, coord_input):
    
    if task_selector == "SAM Mask Generation": ### mask_func.py
        input_points = None
        if coord_input is not None:
            try:
                x, y = map(int, coord_input.split(','))
                input_points = [[[x, y]]]
            except ValueError:
                print("Invalid input format for coordinates (expected: x,y)")
        # image = image["composite"]
        input_image = input_image.convert("RGB")
        output = get_mask(input_image, input_points)
        image_array = np.where(output, 255, 0).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        return pil_image

    if task_selector == "ControlNet Inpainting": ### inpaint_func.py
        input_image = input_image.resize((512, 512))
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
        return output


if __name__ == "__main__":

    block = gr.Blocks()

    with block:
        with gr.Row():
            with gr.Column():

                task_selector = gr.Dropdown(["SAM Mask Generation", "ControlNet Inpainting"], value="SAM Mask Generation")
                input_image = gr.Image(label="Raw Input Image", sources='upload', type="pil", value="inputs/batman.jpg")
                coord_input = gr.Textbox(label="Pixel Coordinates (x,y)", value="350,500") # for SAM
                mask_image = gr.Image(label="Input Mask (Optional)", sources='upload', type="pil", value="inputs/batman_mask.jpg")
                text_input = gr.Textbox(label="Text Prompt", value="") # for inpainting
                
            with gr.Column():
                # gallery = gr.Gallery(
                # label="Generated images", 
                # show_label=False, 
                # elem_id="gallery", 
                # preview=True, 
                # object_fit="scale-down"
                # )
                output_image = gr.Image(label="Generated Image", type="pil")
                generate_button = gr.Button("Generate")

        generate_button.click(
                fn = run_afm_app,
                inputs=[task_selector, input_image, mask_image, text_input, coord_input],
                outputs = output_image
        )  

    block.launch(share=True)