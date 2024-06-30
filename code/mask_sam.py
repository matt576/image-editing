import torch
from PIL import Image
from transformers import SamModel, SamProcessor
import numpy as np



device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


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



if __name__ == "__main__":

    # 2D location of a window in the image: [x,y] coordinates with (0,0) in the top left corner -> pixels
    input_points = [[[600, 800]]]
    raw_image = Image.open("inputs/eval/eval_1.jpg").convert("RGB")
    image = np.array(raw_image)

    output = get_mask(raw_image, input_points)
    image_array = np.where(output, 255, 0).astype(np.uint8)  # Create a new NumPy array for the current channel
    pil_image = Image.fromarray(image_array)  # Convert NumPy array to PIL image
    pil_image.save(f"outputs/sam/dog-mask.png")  # Save the image
