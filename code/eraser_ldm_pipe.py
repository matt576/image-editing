import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import numpy as np
import argparse, os, sys, glob
from omegaconf import OmegaConf
from tqdm import tqdm

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(os.path.join(parent_dir, 'latent-diffusion'))

# from main import instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler

from operations_image import expand_white_areas

"""
Use sam for mask generation, the do mask dilation, then finally use ldm for object removal.
"""

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

def make_batch_gradio(image, mask, device):
    image = np.array(image.convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch

def ldm_removal_pipe_gradio(input_image, coord_input_text, ddim_steps_pipe):
    
    output_dir_mask = "outputs/sam"
    filename = "mask_gradio_ldm_pipe.png"
    filename_dilated = "mask_gradio_ldm_pipe_dilated.png"

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

    image = input_image

    original_width, original_height = image.size

    if original_width > original_height:
        image = image.resize((768, 512))
    elif original_width < original_height:
        image = image.resize((512, 768))
    else:
        image = image.resize((512, 512))

    mask = pil_mask_dilated

    mask_original_width, mask_original_height = mask.size

    if mask_original_width > mask_original_height:
        mask = mask.resize((768, 512))
    elif mask_original_width < mask_original_height:
        mask = mask.resize((512, 768))
    else:
        mask = mask.resize((512, 512))

    steps = int(ddim_steps_pipe)

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
    
            batch = make_batch_gradio(image, mask, device=device)

            c = model.cond_stage_model.encode(batch["masked_image"])
            cc = torch.nn.functional.interpolate(batch["mask"],
                                                size=c.shape[-2:])
            c = torch.cat((c, cc), dim=1)

            shape = (c.shape[1]-1,)+c.shape[2:]
            samples_ddim, _ = sampler.sample(S=steps, ## changed
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
            inpainted = inpainted.resize((original_width, original_height))
            output_dir = "outputs/gradio"
            filename = "ldm_removal_pipe.png"
            inpainted.save(f"{output_dir}/{filename}")
            return inpainted