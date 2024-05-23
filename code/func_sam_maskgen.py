import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
# print("masks:", masks)
# print("scores:", scores)

masks = masks[0].squeeze(0)
arrays = masks.numpy()

for i, bool_array in enumerate(arrays):
    image_array = np.where(bool_array, 255, 0).astype(np.uint8)  # Create a new NumPy array for the current channel
    pil_image = Image.fromarray(image_array)      # Convert NumPy array to PIL image
    pil_image.save(f"outputs/sam/mask-{i}.png")    # Save the image
