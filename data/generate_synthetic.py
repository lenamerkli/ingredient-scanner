from json import load as load_json, dump as dump_json
from PIL import Image, ImageOps, ImageEnhance

import os
import copy
import math
import random
import numpy as np

from video_to_frames import relative_path

BACKGROUNDS = [_file for _file in os.listdir(relative_path('background')) if _file.endswith('.jpg')]
IMAGE_SIZE = (720, 1280)
REPETITIONS = 2
SEED = 2024 - 7 - 28


def get_bounding_box(data):
    xs = [v["x"] for key1 in data for key2, v in data[key1].items()]
    ys = [v["y"] for key1 in data for key2, v in data[key1].items()]
    return min(xs), min(ys), max(xs), max(ys)


def adjust_coordinates(data, offset_x, offset_y, scaling_factor_x, scaling_factor_y):
    adjusted_data = {}
    for key1, key2_points in data.items():
        adjusted_data[key1] = {}
        for key2, point in key2_points.items():
            adjusted_data[key1][key2] = {
                "x": int((point["x"] - offset_x) * scaling_factor_x),
                "y": int((point["y"] - offset_y) * scaling_factor_y)
            }
    return adjusted_data


def create_zoomed_image(image, data, zoom_factor=1.2):
    original_width, original_height = image.size
    min_x, min_y, max_x, max_y = get_bounding_box(data)

    # Calculate the zoomed region to ensure all points are included
    width = max_x - min_x
    height = max_y - min_y

    zoomed_width = int(width * zoom_factor)
    zoomed_height = int(height * zoom_factor)

    # Ensure the zoomed region is within the original image boundaries
    x0 = max(0, min_x - (zoomed_width - width) // 2)
    y0 = max(0, min_y - (zoomed_height - height) // 2)

    x1 = x0 + zoomed_width
    y1 = y0 + zoomed_height

    # Correct x1 and y1 to be within image bounds
    if x1 > original_width:
        x0 -= x1 - original_width
        x1 = original_width
    if y1 > original_height:
        y0 -= y1 - original_height
        y1 = original_height

    x0 = max(0, x0)
    y0 = max(0, y0)

    # Crop and resize to original dimensions
    cropped_image = image.crop((x0, y0, x1, y1))
    zoomed_image = cropped_image.resize((original_width, original_height), Image.Resampling.LANCZOS)

    # Calculate scaling factors
    scaling_factor_x = original_width / (x1 - x0)
    scaling_factor_y = original_height / (y1 - y0)

    # Adjust coordinates to new region
    adjusted_data = adjust_coordinates(data, x0, y0, scaling_factor_x, scaling_factor_y)

    return zoomed_image, adjusted_data


def add_gaussian_noise(image, mean=0, sigma=1):
    # Convert the image to a NumPy array
    np_image = np.array(image)

    # Add Gaussian noise to the image
    gaussian_noise = np.random.normal(mean, sigma, np_image.shape)
    noisy_image = np_image + gaussian_noise

    # Clip (limit) the values of the noisy image to be within the valid range
    noisy_image = np.clip(noisy_image, 0, 255)

    # Convert the noisy NumPy array back to a PIL Image
    noisy_pil_image = Image.fromarray(noisy_image.astype(np.uint8))

    return noisy_pil_image


def rotate_image(image, data) -> tuple:
    image = image.rotate(180)
    width, height = image.size
    old_data = data.copy()
    data = {
        'top': {
            'left': {
                'x': width - old_data['bottom']['right']['x'],
                'y': height - old_data['bottom']['right']['y'],
            },
            'right': {
                'x': width - old_data['bottom']['left']['x'],
                'y': height - old_data['bottom']['left']['y'],
            },
        },
        'bottom': {
            'left': {
                'x': width - old_data['top']['right']['x'],
                'y': height - old_data['top']['right']['y'],
            },
            'right': {
                'x': width - old_data['top']['left']['x'],
                'y': height - old_data['top']['left']['y'],
            },
        },
        'curvature': {
            'top': {
                'x': width - old_data['curvature']['bottom']['x'],
                'y': height - old_data['curvature']['bottom']['y'],
            },
            'bottom': {
                'x': width - old_data['curvature']['top']['x'],
                'y': height - old_data['curvature']['top']['y'],
            },
        },
    }
    return image, data


def adjust_brightness(image, min_factor=0.5, max_factor=1.5):
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(min_factor, max_factor)
    return enhancer.enhance(factor)


def adjust_contrast(image, min_factor=0.5, max_factor=1.5):
    enhancer = ImageEnhance.Contrast(image)
    factor = random.uniform(min_factor, max_factor)
    return enhancer.enhance(factor)


def apply_background(image, data, max_zoom, max_rotate):
    image = image.convert('RGBA')
    # Load a random background image
    background_path = random.choice(BACKGROUNDS)
    background = Image.open(relative_path(os.path.join('background', background_path)))

    # Resize the background to match IMAGE_SIZE if it's not already the same size
    if background.size != IMAGE_SIZE:
        background = background.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    # Ensure the background is in RGBA if the image has transparency
    if image.mode == 'RGBA':
        background = background.convert('RGBA')

    zoom_factor = random.uniform(1.5, max_zoom)

    # Apply zoom on image and data
    new_size = (int(image.width / zoom_factor), int(image.height / zoom_factor))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    for key1 in data:
        for key2 in data[key1]:
            data[key1][key2]['x'] = int(data[key1][key2]['x'] / zoom_factor)
            data[key1][key2]['y'] = int(data[key1][key2]['y'] / zoom_factor)

    # Rotate the image and data
    rotation_angle = random.uniform(-max_rotate, max_rotate)

    def rotate_point(x, y, image_, rotation_angle_):
        # Convert angle from degrees to radians
        theta = math.radians(rotation_angle_)

        # Original image dimensions
        width, height = image_.size

        # Compute original center
        cx, cy = width / 2, height / 2

        # Translate point to center-based coordinates
        x_c, y_c = x - cx, y - cy

        # Rotate the coordinates
        x_c_rotated = x_c * math.cos(-theta) - y_c * math.sin(-theta)
        y_c_rotated = x_c * math.sin(-theta) + y_c * math.cos(-theta)

        # Calculate the new image size if expand=True
        new_width = abs(width * math.cos(theta)) + abs(height * math.sin(theta))
        new_height = abs(width * math.sin(theta)) + abs(height * math.cos(theta))

        # Compute new center
        cx_prime, cy_prime = new_width / 2, new_height / 2

        # Translate back to top-left based coordinates
        x_prime = x_c_rotated + cx_prime
        y_prime = y_c_rotated + cy_prime

        return x_prime, y_prime

    for key1 in data:
        for key2 in data[key1]:
            data[key1][key2]['x'], data[key1][key2]['y'] = rotate_point(data[key1][key2]['x'], data[key1][key2]['y'],
                                                                        image, rotation_angle)
    image = image.rotate(rotation_angle, expand=True)

    # Calculate the maximum position to place the image such that it's fully visible
    max_x = IMAGE_SIZE[0] - image.size[0]
    max_y = IMAGE_SIZE[1] - image.size[1]

    # Choose a random position within the allowed range
    pos_x = random.randint(0, max(max_x, 0))
    pos_y = random.randint(0, max(max_y, 0))

    if image.mode == 'RGBA':
        # Use the alpha channel as the mask for transparency
        mask = image.split()[3]
        background.paste(image, (pos_x, pos_y), mask)
    else:
        background.paste(image, (pos_x, pos_y))

    for key1 in data:
        for key2 in data[key1]:
            data[key1][key2]['x'] = data[key1][key2]['x'] + pos_x
            data[key1][key2]['y'] = data[key1][key2]['y'] + pos_y

    if background.mode == 'RGBA':
        background = background.convert('RGB')

    return background, data


def main():
    """
    Generates synthetic_frames data from the 'frames' and 'frames_json' directory
    and saves it in the 'synthetic_frames' directory.
    :return: None
    """
    num_settings = 4
    random.seed(SEED)
    for file in sorted(os.listdir(relative_path('frames_json'))):
        if file.split('.')[-1].strip().lower() == 'json':
            with open(relative_path(f"frames_json/{file}"), 'r') as f:
                original_data: dict = load_json(f)
            for frame in sorted(os.listdir(relative_path('frames'))):
                if frame.startswith(file.split('.')[0]):
                    if original_data['curvature']['top']['x'] is None:
                        original_data['curvature']['top']['x'] = (original_data['top']['left']['x']
                                                                  + original_data['top']['right']['x']) / 2
                    if original_data['curvature']['top']['y'] is None:
                        original_data['curvature']['top']['y'] = (original_data['top']['left']['y']
                                                                  + original_data['top']['right']['y']) / 2
                    if original_data['curvature']['bottom']['x'] is None:
                        original_data['curvature']['bottom']['x'] = (original_data['bottom']['left']['x']
                                                                     + original_data['bottom']['right']['x']) / 2
                    if original_data['curvature']['bottom']['y'] is None:
                        original_data['curvature']['bottom']['y'] = (original_data['bottom']['left']['y']
                                                                     + original_data['bottom']['right']['y']) / 2
                    with Image.open(relative_path(f"frames/{frame}")) as original_image:
                        index = 0
                        for repetition in range(REPETITIONS):
                            for settings_int in range(2 ** num_settings):
                                settings = bin(settings_int)[2:].zfill(num_settings)
                                if settings.count('1') >= 1:
                                    data = copy.deepcopy(original_data)
                                    image = original_image.copy()
                                    image = add_gaussian_noise(image)
                                    image = adjust_brightness(image)
                                    image = adjust_contrast(image)
                                    if settings[0] == '1':
                                        image, data = rotate_image(image, data)
                                    if settings[1] == '1':
                                        image = ImageOps.invert(image)
                                    if settings[2] == '1' or settings[3] == '1':
                                        image, data = apply_background(image, data, 2, 35)
                                    image.save(relative_path(f"synthetic_frames/{frame.split('.')[0]}_{index:03}.png"))
                                    with open(relative_path(f"synthetic_frames_json/{frame.split('.')[0]}"
                                                            f"_{index:03}.json"), 'w') as f:
                                        dump_json(data, f, indent=2)
                                    del data
                                    del image
                                    print(f"Image: {frame.split('.')[0]}, Settings: {settings}, Index: {index}")
                                    index += 1
                    break


if __name__ == '__main__':
    main()
