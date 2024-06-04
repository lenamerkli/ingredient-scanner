from os import listdir
from tqdm import tqdm
from json import load as load_json, dump as dump_json
from PIL import Image

import random

from video_to_frames import relative_path


REPETITIONS = 8
SEED = 2024-7-28


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
                    max_floats = {
                        'top': min(data['top']['left']['y'], data['top']['right']['y'],
                                   data['curvature']['top']['y']),
                        'bottom': max(data['bottom']['left']['y'], data['bottom']['right']['y'],
                                      data['curvature']['bottom']['y']),
                        'left': min(data['top']['left']['x'], data['bottom']['left']['x']),
                        'right': max(data['top']['right']['x'], data['bottom']['right']['x']),
                    }
                    with Image.open(relative_path(f"frames/{frame}")) as image:
                        width, height = image.size[:2]
                        for repetition in range(REPETITIONS):
                            top = random.randint(0, max_floats['top'])
                            bottom = random.randint(0, max_floats['bottom'])
                            left = random.randint(0, max_floats['left'])
                            right = random.randint(0, max_floats['right'])
                            cropped_image = image.crop((left, top, width - right, height - bottom))
                            output_image = cropped_image.resize((width, height))
                            output_image.save(relative_path(f"synthetic_frames/{frame.split('.')[0]}_{repetition:03}.png"))
                            output_data = {
                                'top': {
                                    'left': {
                                        'x': int(data['top']['left']['x'] - left),
                                        'y': int(data['top']['left']['y'] - top),
                                    },
                                    'right': {
                                        'x': int(data['top']['right']['x'] + right),
                                        'y': int(data['top']['right']['y'] - top),
                                    },
                                },
                                'bottom': {
                                    'left': {
                                        'x': int(data['bottom']['left']['x'] - left),
                                        'y': int(data['bottom']['left']['y'] + bottom),
                                    },
                                    'right': {
                                        'x': int(data['bottom']['right']['x'] + right),
                                        'y': int(data['bottom']['right']['y'] + bottom),
                                    },
                                },
                                'curvature': {
                                    'top': {
                                        'x': int(data['curvature']['top']['x'] - left + right),
                                        'y': int(data['curvature']['top']['y'] - top),
                                    },
                                    'bottom': {
                                        'x': int(data['curvature']['bottom']['x'] - left + right),
                                        'y': int(data['curvature']['bottom']['y'] + bottom),
                                    },
                                },
                            }
                            with open(relative_path(f"synthetic_frames_json/{frame.split('.')[0]}_{repetition:03}.json"),
                                      'w') as f:
                                dump_json(output_data, f, indent=2)
                    break


if __name__ == '__main__':
    main()
