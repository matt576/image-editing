import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import numpy as np

output_dir = "outputs/sam"
filename = "car-mask-2.png"

# 2D location of a window in the image: [x,y] coordinates with (0,0) in the top left corner -> pixels
input_points = [[[450, 600], [900, 600]]]
raw_image = Image.open("inputs/examples_depth/car.png").convert("RGB")
image = np.array(raw_image)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")


def get_mask(input_image, input_points):
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

    # arrays = masks.numpy()
    # for i, bool_array in enumerate(arrays):
    #     image_array = np.where(bool_array, 255, 0).astype(np.uint8)  # Create a new NumPy array for the current channel
    #     pil_image = Image.fromarray(image_array)  # Convert NumPy array to PIL image
    #     pil_image.save(f"outputs/sam/mask-{i}.png")  # Save the image


output = get_mask(raw_image, input_points)
image_array = np.where(output, 255, 0).astype(np.uint8)  # Create a new NumPy array for the current channel
pil_image = Image.fromarray(image_array)  # Convert NumPy array to PIL image
pil_image.save(f"{output_dir}/{filename}")  # Save the image
