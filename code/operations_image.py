from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion



# function taking RGB image as input, a transforming it to a black-white image basing on the pixel brightness
def rgb_to_thresholded_grayscale(input_image, threshold=127):
    
    img = input_image.convert('L')  # Convert to grayscale
    thresholded_img = img.point(lambda p: 255 if p >= threshold else 0)
    return thresholded_img


# Function to resize an image
def read_and_resize_image(input_image, new_size):

    resized_img = input_image.resize(new_size)
    return resized_img


# Expand white areas of an image
def expand_white_areas(input_image, iterations):
    
    img_array = np.array(input_image)

    # Create a binary mask where white pixels are True and black pixels are False and expand white areas
    binary_mask = img_array == 255
    dilated_mask = binary_dilation(binary_mask, iterations=iterations)
    expanded_array = np.where(dilated_mask, 255, 0).astype(np.uint8)
    expanded_img = Image.fromarray(expanded_array, mode='L')
    return expanded_img


# shrink white areas of an image
def shrink_white_areas(mask_image, iterations=1):

    img = mask_image.convert('L')
    img_array = np.array(img)
    binary_mask = img_array == 255
    eroded_mask = binary_erosion(binary_mask, iterations=iterations)
    shrunk_array = np.where(eroded_mask, 255, 0).astype(np.uint8)
    shrunk_img = Image.fromarray(shrunk_array, mode='L')
    return shrunk_img


if __name__ == "__main__":

    # resize image
    input_path = 'inputs/sleep.png'    # Path to the input image
    output_path = 'inputs/inpainting/sleep.png' # Path to save the resized image
    new_size = (480, 640)                    # New size as a tuple (width, height)
    input_img = Image.open(input_path).convert("RGB")
    output_img = read_and_resize_image(input_img, new_size)
    output_img.save(output_path)
