import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import numpy as np


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

    # arrays = masks.numpy()
    # for i, bool_array in enumerate(arrays):
    #     image_array = np.where(bool_array, 255, 0).astype(np.uint8)  # Create a new NumPy array for the current channel
    #     pil_image = Image.fromarray(image_array)  # Convert NumPy array to PIL image
    #     pil_image.save(f"outputs/sam/mask-{i}.png")  # Save the image

if __name__ == "__main__":

    output_dir = "outputs/sam"
    filename = "car-mask-2.png"

    # 2D location of a window in the image: [x,y] coordinates with (0,0) in the top left corner -> pixels
    input_points = [[[515,575],[803,558],[1684,841]]]
    raw_image = Image.open("inputs/car.png").convert("RGB")
    image = np.array(raw_image)

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    # processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    # processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    # img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    
    output = get_mask(raw_image, input_points)
    image_array = np.where(output, 255, 0).astype(np.uint8)  # Create a new NumPy array for the current channel
    pil_image = Image.fromarray(image_array)  # Convert NumPy array to PIL image
    pil_image.save(f"{output_dir}/{filename}")  # Save the image


# def sam_gradio(input_image, coord_input): # for one point x,y
#     output_dir = "outputs/sam"
#     filename = "mask_gradio.png"
#     input_points = None

#     if coord_input is not None:
#         try:
#             x, y = map(int, coord_input.split(','))
#             input_points = [[[x, y]]]
#         except ValueError:
#             print("Invalid input format for coordinates (expected: x,y)")

#     # image = image["composite"]
#     input_image = input_image.convert("RGB")
#     output = get_mask(input_image, input_points)
#     image_array = np.where(output, 255, 0).astype(np.uint8)
#     pil_image = Image.fromarray(image_array)
#     pil_image.save(f"{output_dir}/{filename}") 
#     return pil_image

def sam_gradio(input_image, coord_input_text):
    output_dir = "outputs/sam"
    filename = "mask_gradio.png"

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
    output = get_mask(input_image, input_points)
    image_array = np.where(output, 255, 0).astype(np.uint8)
    pil_image = Image.fromarray(image_array)
    pil_image.save(f"{output_dir}/{filename}") 
    return pil_image