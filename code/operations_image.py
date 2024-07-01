from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation


# Function to read and resize an image
def read_and_resize_image(input_image, new_size):

    # Resize the image
    resized_img = input_image.resize(new_size)
    
    # Save the resized image to the output path
    resized_img.save(output_path)
    return resized_img


def expand_white_areas(image_path, iterations):
    
    # Load the image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    print("iterations: ", iterations)
    # Create a binary mask where white pixels are True and black pixels are False and expand white areas
    binary_mask = img_array == 255
    dilated_mask = binary_dilation(binary_mask, iterations=iterations)
    expanded_array = np.where(dilated_mask, 255, 0).astype(np.uint8)
    expanded_img = Image.fromarray(expanded_array, mode='L')
    return expanded_img


"""
# Example usage
input_path = 'inputs/eval/eval_1_mask.png'    # Path to the input image
output_path = 'inputs/eval/eval_2_mask.png' # Path to save the resized image
new_size = (480, 640)                    # New size as a tuple (width, height)
read_and_resize_image(input_path, output_path, new_size)

# Example usage
dil_iterations = 1001
image_path = 'inputs/eval/eval_2_mask.png'
image_input = Image.open(image_path)
image_output = expand_white_areas(image_path, iterations=dil_iterations)
output_path = f'inputs/eval/eval_2_mask_d{dil_iterations}.png'
image_output.save(output_path)
"""

if __name__ == "__main__":
    
    dil_iterations = 10
    image_path = 'outputs/sam/mask_gradio.png'
    image_input = Image.open(image_path)
    image_output = expand_white_areas(image_path, iterations=dil_iterations)
    output_path = f'inputs/eval/eval_2_mask_d{dil_iterations}.png'
    image_output.save(output_path)