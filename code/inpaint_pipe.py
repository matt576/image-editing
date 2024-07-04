import os, sys
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import numpy as np
import argparse, os, sys, glob
from omegaconf import OmegaConf
from tqdm import tqdm

from diffusers import AutoPipelineForInpainting

from operations_image import expand_white_areas

def get_mask(input_image, input_points):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    inputs = processor(input_image, input_points=input_points, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    scores = outputs.iou_scores
    masks = masks[0].squeeze(0)
    max_index = torch.argmax(scores)
    best_mask = masks[max_index]
    return best_mask

def inpaint_pipe_gradio(task_selector, input_image, coord_input_text, text_input_inpaint_pipe, np_inpaint, steps_inpaint):
    print("task_selector: ", task_selector)
    prompt = text_input_inpaint_pipe
    output_dir_mask = "outputs/sam"
    filename = "mask_gradio_inpaint_pipe.png"
    filename_dilated = "mask_gradio_inpaint_pipe_dilated.png"
    steps_inpaint = int(steps_inpaint)
    input_points = None
    if coord_input_text is not None:
        try:
            points = coord_input_text.split(';')
            input_points = []
            for point in points:
                x, y = map(int, point.split(',')) # Split by comma to get x and y coordinates
                input_points.append([x, y])
            input_points = [input_points] # Wrap input_points in another list to match the expected format e.g. [[[515,575],[803,558],[1684,841]]]
        except ValueError:
            print("Invalid input format for coordinates (expected: x1,y1;x2,y2;x3,y3)")
            input_points = None

    input_image = input_image.convert("RGB")
    output_mask = get_mask(input_image, input_points)
    image_array = np.where(output_mask, 255, 0).astype(np.uint8)
    pil_mask = Image.fromarray(image_array)
    # pil_mask.save(f"{output_dir_mask}/{filename}") 
    
    image_path = f"{output_dir_mask}/{filename}"
    dil_iterations = 10
    pil_mask_dilated = expand_white_areas(image_path, iterations=dil_iterations)
    pil_mask_dilated.save(f"{output_dir_mask}/{filename_dilated}")

    input_image = input_image.convert("RGB")
    original_width, original_height = input_image.size
    input_image = input_image.resize((512, 512))

    mask_image = pil_mask_dilated
    mask_image = mask_image.convert("L")
    mask_image = mask_image.resize(input_image.size)

    if task_selector == "Stable Diffusion v1.5 Inpainting Pipeline":
        pipeline = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        variant="fp16",
        torch_dtype=torch.float16,
        )
        pipeline.enable_model_cpu_offload()
        print(prompt)
        image = pipeline(prompt=prompt, image=input_image, mask_image=mask_image, negative_prompt=np_inpaint,num_inference_steps=steps_inpaint).images[0]
        
        image_resized = image.resize((original_width, original_height))
        output_dir = "outputs/gradio/inpainting"
        filename = "inpaint_pipe_sd_output.png"
        image_resized.save(f"{output_dir}/{filename}")
        return image_resized 

    elif task_selector == "Stable Diffusion XL Inpainting Pipeline":

        pipeline = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
        )
        pipeline.enable_model_cpu_offload()

        generator = torch.Generator("cuda").manual_seed(92)

        # mask_image = pipeline.mask_processor.blur(mask_image, blur_factor=33) #Optional
        print(prompt)
        image = pipeline(prompt=prompt, image=input_image, mask_image=mask_image, generator=generator, negative_prompt=np_inpaint,num_inference_steps=steps_inpaint).images[0]
        
        image_resized = image.resize((original_width, original_height))
        output_dir = "outputs/gradio/inpainting"
        filename = "inpaint_pipe_sdxl_output.png"
        image_resized.save(f"{output_dir}/{filename}")
        return image_resized 

    elif task_selector == "Kandinsky v2.2 Inpainting Pipeline":
        pipeline = AutoPipelineForInpainting.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
        )
        pipeline.enable_model_cpu_offload()

        generator = torch.Generator("cuda").manual_seed(92)
        print(prompt)
        image = pipeline(prompt=prompt, image=input_image, mask_image=mask_image, generator=generator, negative_prompt=np_inpaint,num_inference_steps=steps_inpaint).images[0]
        
        image_resized = image.resize((original_width, original_height))
        output_dir = "outputs/gradio/inpainting"
        filename = "inpaint_pipe_kandinsky_output.png"
        image_resized.save(f"{output_dir}/{filename}")
        return image_resized

    else:
        print("Please select a valid model.")