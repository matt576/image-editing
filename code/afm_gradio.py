import gradio as gr
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import numpy as np

import os, sys

import torchvision.transforms.v2.functional

from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler

# imports from code scripts
from mask_func import get_mask

import gradio as gr
from PIL import Image

# Assuming get_mask is defined in a separate script (mask_func.py)
from mask_func import get_mask

# Pre-defined input points for mask generation (adjust as needed)
input_points = [[[350, 500]]]

def generate_mask(image):
  image = image.convert("RGB")
  output = get_mask(image, input_points)
  image_array = np.where(output, 255, 0).astype(np.uint8)
  pil_image = Image.fromarray(image_array)
  return pil_image

with gr.Blocks() as demo:
  with gr.Row():
    # Image upload for mask generation
    uploaded_image = gr.Image(label="Input Image", type="pil")
    masked_image = gr.Image(label="Masked Image", type="pil")
    generate_button = gr.Button("Generate Mask")

  # Link button click to generate_mask function
  generate_button.click(fn=generate_mask, inputs=uploaded_image, outputs=masked_image)

# Launch the Gradio app
demo.launch(share=True)

# def update(name):
#     return f"Welcome to Gradio, {name}!"

# with gr.Blocks() as demo:
#     gr.Markdown("Start typing below and then click **Run** to see the output.")
#     with gr.Row():
#         inp = gr.Textbox(placeholder="What is your name?")
#         out = gr.Textbox()
#     btn = gr.Button("Run")
#     btn.click(fn=update, inputs=inp, outputs=out)

# demo.launch()