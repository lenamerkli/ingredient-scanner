from os import listdir
from tqdm import tqdm
from json import load as load_json, dump as dump_json
from PIL import Image, ImageFilter

import random

from video_to_frames import relative_path

REPETITIONS = 16
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

    # Ensure the zoomed region doesn't exceed the original image boundaries
    x0 = max(0, min_x - random.randint(0, zoomed_width - width))
    y0 = max(0, min_y - random.randint(0, zoomed_height - height))
    x1 = min(original_width, x0 + zoomed_width)
    y1 = min(original_height, y0 + zoomed_height)

    # Crop and resize to original dimensions
    cropped_image = image.crop((x0, y0, x1, y1))
    zoomed_image = cropped_image.resize((original_width, original_height), Image.Resampling.LANCZOS)

    # Calculate scaling factors
    scaling_factor_x = original_width / (x1 - x0)
    scaling_factor_y = original_height / (y1 - y0)

    # Adjust coordinates to new region
    adjusted_data = adjust_coordinates(data, x0, y0, scaling_factor_x, scaling_factor_y)

    return zoomed_image, adjusted_data


def main():
    """
    Generates synthetic_frames data from the 'frames' and 'frames_json' directory
    and saves it in the 'synthetic_frames' directory.
    :return: None
    """
    random.seed(SEED)
    for file in tqdm(sorted(listdir(relative_path('frames_json')))):
        if file.split('.')[-1].strip().lower() == 'json':
            with open(relative_path(f"frames_json/{file}"), 'r') as f:
                data = load_json(f)
            for frame in sorted(listdir(relative_path('frames'))):
                if frame.startswith(file.split('.')[0]):
                    if data['curvature']['top']['x'] is None:
                        data['curvature']['top']['x'] = (data['top']['left']['x'] + data['top']['right']['x']) / 2
                    if data['curvature']['top']['y'] is None:
                        data['curvature']['top']['y'] = (data['top']['left']['y'] + data['top']['right']['y']) / 2
                    if data['curvature']['bottom']['x'] is None:
                        data['curvature']['bottom']['x'] = (data['bottom']['left']['x'] + data['bottom']['right']['x']) / 2
                    if data['curvature']['bottom']['y'] is None:
                        data['curvature']['bottom']['y'] = (data['bottom']['left']['y'] + data['bottom']['right']['y']) / 2
                    with Image.open(relative_path(f"frames/{frame}")) as image:
                        for repetition in range(REPETITIONS):
                            noised_image = image.filter(ImageFilter.GaussianBlur(4))
                            zoomed_image, adjusted_data = create_zoomed_image(image, data)
                            if repetition % 2 == 0:
                                zoomed_image = zoomed_image.rotate(180)
                                width, height = zoomed_image.size
                                old_data = adjusted_data.copy()
                                adjusted_data = {
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
                            zoomed_image.save(relative_path(f"synthetic_frames/{frame.split('.')[0]}_{repetition:03}.png"))
                            with open(relative_path(f"synthetic_frames_json/{frame.split('.')[0]}_{repetition:03}.json"),
                                      'w') as f:
                                dump_json(adjusted_data, f, indent=2)
                    break


if __name__ == '__main__':
    main()
