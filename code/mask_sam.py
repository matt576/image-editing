import torch
from PIL import Image
from transformers import SamModel, SamProcessor
import numpy as np
from operations_image import expand_white_areas_outpainting


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


def sam_gradio(input_image, coord_input_text, dilation_bool, dilation_value):
    dilation_value = int(dilation_value)
    output_dir = "outputs/sam"
    # output_dir = "code/inputs/gradio_masks"
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

    if dilation_bool == "Yes":
        pil_image = expand_white_areas_outpainting(pil_image, dilation_value)

    pil_image.save(f"{output_dir}/{filename}")
    return pil_image


if __name__ == "__main__":

    output_dir = "outputs/sam"
    filename = "car-mask-2.png"

    # 2D location of a window in the image: [x,y] coordinates with (0,0) in the top left corner -> pixels
    input_points = [[[515,575],[803,558],[1684,841]]]
    raw_image = Image.open("inputs/car.png").convert("RGB")
    image = np.array(raw_image)
    output = get_mask(raw_image, input_points)
    image_array = np.where(output, 255, 0).astype(np.uint8)  # Create a new NumPy array for the current channel
    pil_image = Image.fromarray(image_array)  # Convert NumPy array to PIL image
    pil_image.save(f"{output_dir}/{filename}")  # Save the image
