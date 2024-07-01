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

from transformers import SamModel, SamProcessor

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

def inpaint_func_pipe_gradio(input_image, coord_input_text, text_input):
    output_dir_mask = "outputs/sam"
    filename = "mask_gradio_inpaint_pipe.png"
    filename_dilated = "mask_gradio_inpaint_pipe_dilated.png"

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
    pil_mask.save(f"{output_dir_mask}/{filename}") 
    
    image_path = f"{output_dir_mask}/{filename}"
    dil_iterations = 10
    pil_mask_dilated = expand_white_areas(image_path, iterations=dil_iterations)
    pil_mask_dilated.save(f"{output_dir_mask}/{filename_dilated}")

    mask_image = pil_mask_dilated  

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
    output_dir = "outputs/gradio/inpainting"
    filename = "sdv15controlnet_inpaint_func_pipe_output.png"
    output.save(f"{output_dir}/{filename}")
    return output