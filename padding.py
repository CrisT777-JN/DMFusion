import os
from PIL import Image
import math

def resize_to_multiple_of_32(img):
    width, height = img.size
    new_width = math.ceil(width / 32) * 32
    new_height = math.ceil(height / 32) * 32
    return img.resize((new_width, new_height), Image.BICUBIC)

def process_images(input_folder, output_folder=None):
    if output_folder is None:
        output_folder = input_folder  
    else:
        os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            path = os.path.join(input_folder, filename)
            img = Image.open(path)
            resized_img = resize_to_multiple_of_32(img)
            save_path = os.path.join(output_folder, filename)
            resized_img.save(save_path)
            print(f"Processed: {filename} -> {resized_img.size}")

input_dir = 'test_data/LLVIP/Vis'  
process_images(input_dir)
