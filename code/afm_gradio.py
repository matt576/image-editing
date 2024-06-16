import gradio as gr
import torch
import os, sys
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import torchvision.transforms.v2.functional
from transformers import SamModel, SamProcessor
from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline, LDMSuperResolutionPipeline

# imports from code scripts
from mask_func import sam_gradio
from inpaint_func import inputation #, make_inpaint_condition, controlnet, pipe
from inpaint_ldm import make_batch
from superres_func import superres_gradio
from restyling_func import restyling_gradio

def run_afm_app(task_selector, input_image, mask_image, text_input, coord_input, ddim_steps):
    
    if task_selector == "SAM Mask Generation": ### mask_func.py
        return sam_gradio(input_image, coord_input)
        # input_points = None
        # if coord_input is not None:
        #     try:
        #         x, y = map(int, coord_input.split(','))
        #         input_points = [[[x, y]]]
        #     except ValueError:
        #         print("Invalid input format for coordinates (expected: x,y)")
        # # image = image["composite"]
        # input_image = input_image.convert("RGB")
        # output = get_mask(input_image, input_points)
        # image_array = np.where(output, 255, 0).astype(np.uint8)
        # pil_image = Image.fromarray(image_array)
        # return pil_image

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

    if task_selector == "Object Removal":
        image = input_image
        image = image.resize((512, 512))
        mask = mask_image
        mask = mask.resize((512, 512))
        steps = int(ddim_steps)

        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        sys.path.append(os.path.join(parent_dir, 'latent-diffusion'))
        from main import instantiate_from_config
        from ldm.models.diffusion.ddim import DDIMSampler
        
        config = OmegaConf.load("models/ldm_inpainting/config.yaml")
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load("models/ldm_inpainting/last.ckpt")["state_dict"],
                            strict=False)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        sampler = DDIMSampler(model)

        with torch.no_grad():
            with model.ema_scope():
        
                batch = make_batch(image, mask, device=device)

                # encode masked image and concat downsampled mask
                c = model.cond_stage_model.encode(batch["masked_image"])
                cc = torch.nn.functional.interpolate(batch["mask"],
                                                    size=c.shape[-2:])
                c = torch.cat((c, cc), dim=1)

                shape = (c.shape[1]-1,)+c.shape[2:]
                samples_ddim, _ = sampler.sample(S=steps, ## change
                                                conditioning=c,
                                                batch_size=c.shape[0],
                                                shape=shape,
                                                verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                            min=0.0, max=1.0)

                inpainted = (1-mask)*image+mask*predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                inpainted = Image.fromarray(inpainted.astype(np.uint8))
                return inpainted
    
    if task_selector == "Restyling":
        return restyling_gradio(input_image, text_input)

    if task_selector == "Hyperresolution":
        return superres_gradio(input_image)


if __name__ == "__main__":

    block = gr.Blocks()

    with block:
        with gr.Row():
            with gr.Column():

                task_selector = gr.Dropdown(["SAM Mask Generation", 
                                            "ControlNet Inpainting", 
                                            "Object Removal",
                                            "Restyling",
                                            "Hyperresolution"], 
                                            value="SAM Mask Generation")
                input_image = gr.Image(label="Raw Input Image", sources='upload', type="pil", value="inputs/dog.png")
                coord_input = gr.Textbox(label="Pixel Coordinates (x,y)", value="350,500") # for SAM
                mask_image = gr.Image(label="Input Mask (Optional)", sources='upload', type="pil", value="inputs/dog_mask.png")
                text_input = gr.Textbox(label="Text Prompt", value="") # for inpainting or restyling
                ddim_steps = gr.Textbox(label="Number of DDIM sampling steps for object removal", value="50") # for inpaint_ldm
                
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
                inputs=[task_selector, input_image, mask_image, text_input, coord_input, ddim_steps],
                outputs = output_image
        )  

    block.launch(share=True)