import numpy as np
import torch
import easyocr
import cv2
import os
import base64
import json
import requests
from llama_cpp import Llama
from PIL import Image
from dotenv import load_dotenv

from utils import *

load_dotenv()

SCALE_FACTOR: int | float = 4  # how much the size of the cut-out image should be scaled up
MAX_SIZE: int = 5_000_000  # maximum file size in bytes, using the WEBP format
MAX_SIDE: int = 8_000  # maximum size of any side of the cut-out image in pixels
# ENGINE: list[str] = ['easyocr']
# ENGINE: list[str] = ['anthropic', 'claude-3-5-sonnet-20240620']
ENGINE: list[str] = ['llama_cpp/v2/vision', 'qwen-vl-next_b2583']


def main() -> None:
    """
    Main inference function using user input.
    :return: None
    """
    # load vision model
    model_weights = torch.load(relative_path('vision_model.pt'))
    model = NeuralNet()
    model.load_state_dict(model_weights)
    # move model to the preferred device
    model.to(DEVICE)
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
        file_path = input('Enter file path including .png extension: ')
        with Image.open(file_path) as image:
            image_size = image.size
            # resize the image to match the input size of the model
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            image = TRANSFORM(image).to(DEVICE)
            # make the prediction and convert to dictionary
            output = model(image).tolist()[0]
            data = {
                'top': {
                    'left': {
                        'x': output[0] * image_size[0],
                        'y': output[1] * image_size[1],
                    },
                    'right': {
                        'x': output[2] * image_size[0],
                        'y': output[3] * image_size[1],
                    },
                },
                'bottom': {
                    'left': {
                        'x': output[4] * image_size[0],
                        'y': output[5] * image_size[1],
                    },
                    'right': {
                        'x': output[6] * image_size[0],
                        'y': output[7] * image_size[1],
                    },
                },
                'curvature': {
                    'top': {
                        'x': output[8] * image_size[0],
                        'y': output[9] * image_size[1],
                    },
                    'bottom': {
                        'x': output[10] * image_size[0],
                        'y': output[11] * image_size[1],
                    },
                },
            }
            print(f"{data=}")
    # Warp the image based on the predicted coordinates (from the distort.py script, see comments there)
    image = cv2.imread(file_path)
    size_x = ((data['top']['right']['x'] - data['top']['left']['x']) +
              (data['bottom']['right']['x'] - data['bottom']['left']['x'])) / 2
    size_y = ((data['top']['right']['y'] - data['top']['left']['y']) +
              (data['bottom']['right']['y'] - data['bottom']['left']['y'])) / 2
    margin_x = size_x * MARGIN
    margin_y = size_y * MARGIN
    points = np.array([
        (max(data['top']['left']['x'] - margin_x, 0),
         max(data['top']['left']['y'] - margin_y, 0)),
        (min(data['top']['right']['x'] + margin_x, image_size[0]),
         max(data['top']['right']['y'] - margin_y, 0)),
        (min(data['bottom']['right']['x'] + margin_x, image_size[0]),
         min(data['bottom']['right']['y'] + margin_y, image_size[1])),
        (max(data['bottom']['left']['x'] - margin_x, 0),
         min(data['bottom']['left']['y'] + margin_y, image_size[1])),
        (data['curvature']['top']['x'],
         max(data['curvature']['top']['y'] - margin_y, 0)),
        (data['curvature']['bottom']['x'],
         min(data['curvature']['bottom']['y'] + margin_y, image_size[1])),
    ], dtype=np.float32)
    points_float: list[list[float]] = points.tolist()
    max_height = int(max([  # y: top left - bottom left, top right - bottom right, curvature top - curvature bottom
        abs(points_float[0][1] - points_float[3][1]),
        abs(points_float[1][1] - points_float[4][1]),
        abs(points_float[2][1] - points_float[5][1]),
    ])) * SCALE_FACTOR
    max_width = int(max([  # x: top left - top right, bottom left - bottom right
        abs(points_float[0][0] - points_float[1][0]),
        abs(points_float[3][0] - points_float[2][0]),
    ])) * SCALE_FACTOR
    destination_points = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
        [max_width // 2, 0],
        [max_width // 2, max_height - 1],
    ], dtype=np.float32)
    # https://learnopencv.com/homography-examples-using-opencv-python-c/
    homography, _ = cv2.findHomography(points, destination_points)
    warped_image = cv2.warpPerspective(image, homography, (max_width, max_height))
    cv2.imwrite('_warped_image.png', warped_image)
    del data
    # selection of the OCR engine
    if ENGINE[0] == 'easyocr':
        # https://github.com/JaidedAI/EasyOCR
        reader = easyocr.Reader(['de', 'fr', 'en'], gpu=True)
        result = reader.readtext('_warped_image.png')
        os.remove('_warped_image.png')
        text = '\n'.join([r[1] for r in result])
        ingredients = {}
    elif ENGINE[0] == 'anthropic':
        # decrease size of the image to fit it into the requirements of anthropic
        decrease_size('_warped_image.png', '_warped_image.webp', MAX_SIZE, MAX_SIDE)
        os.remove('_warped_image.png')
        with open('_warped_image.webp', 'rb') as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        # https://docs.anthropic.com/en/api/messages
        response = requests.post(
            url='https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': os.environ['ANTHROPIC_API_KEY'],
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json',
            },
            data=json.dumps({
                'model': ENGINE[1],
                'max_tokens': 1024,
                'messages': [
                    {
                        'role': 'user', 'content': [
                            {
                                'type': 'image',
                                'source': {
                                    'type': 'base64',
                                    'media_type': 'image/webp',
                                    'data': base64_image,
                                },
                            },
                            {
                                'type': 'text',
                                'text': PROMPT_CLAUDE,
                            },
                        ],
                    },
                ],
            }),
        )
        os.remove('_warped_image.webp')
        try:
            data = response.json()
            # extract the JSON data from the response
            ingredients = json.loads('{' + data['content'][0]['text'].split('{', 1)[-1].rsplit('}', 1)[0] + '}')
        except Exception as e:
            print(data)
            raise e
        text = ''
    elif ENGINE[0] == 'llama_cpp/v2/vision':
        # decrease size of the image to fit it into the requirements of llama_cpp
        decrease_size('_warped_image.png', '_warped_image.webp', MAX_SIZE, MAX_SIDE)
        os.remove('_warped_image.png')
        response = requests.post(
            url='http://127.0.0.1:11434/llama_cpp/v2/vision',
            headers={
                'x-version': '2024-05-21',
                'content-type': 'application/json',
            },
            data=json.dumps({
                'task': PROMPT_VISION,
                'model': ENGINE[1],
                'image_path': relative_path('_warped_image.webp'),
            }),
        )
        os.remove('_warped_image.webp')
        text: str = response.json()['text']
        ingredients = {}
    else:
        raise ValueError(f'Unknown engine: {ENGINE[0]}')
    if text != '':
        # only call the LLM if the ingredients could not be extracted from the text
        if DEVICE == 'cuda':
            # if cuda is available, load the entire model to the GPU
            n_gpu_layers = -1
        else:
            n_gpu_layers = 0
        llm = Llama(
            model_path=relative_path('llm.Q4_K_M.gguf'),
            n_gpu_layers=n_gpu_layers,
        )
        llm_result = llm.create_chat_completion(
            messages=[
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT,
                },
                {
                    'role': 'user',
                    'content': PROMPT_LLM.replace('{{old_data}}', text),
                },
            ],
            max_tokens=1024,
            temperature=0,  # no randomness
            # grammar=GRAMMAR,  # grammar disabled due to a bug
        )
        try:
            # extract the JSON data from the response
            ingredients = json.loads(
                '{' + llm_result['choices'][0]['message']['content'].split('{', 1)[-1].rsplit('}', 1)[0] + '}')
        except Exception as e:
            print(f"{llm_result=}")
            raise e
    animal_ingredients = [item for item in ingredients['Zutaten'] if item in ANIMAL]
    sometimes_animal_ingredients = [item for item in ingredients['Zutaten'] if item in SOMETIMES_ANIMAL]
    milk_ingredients = ([item for item in ingredients['Zutaten'] if item in MILK]
                        + [item for item in ingredients['Verunreinigungen'] if item in MILK])
    gluten_ingredients = ([item for item in ingredients['Zutaten'] if item in GLUTEN]
                          + [item for item in ingredients['Verunreinigungen'] if item in GLUTEN])
    print('=' * 64)
    print('Zutaten: ' + ', '.join(ingredients['Zutaten']))
    print('=' * 64)
    print(('Kann Spuren von ' + ', '.join(ingredients['Verunreinigungen']) + ' enthalten.')
          if len(ingredients['Verunreinigungen']) > 0 else 'ohne Verunreinigungen')
    print('=' * 64)
    print('Gefundene tierische Zutaten: '
          + (', '.join(animal_ingredients) if len(animal_ingredients) > 0 else 'keine'))
    print('=' * 64)
    print('Gefundene potenzielle tierische Zutaten: '
          + (', '.join(sometimes_animal_ingredients) if len(sometimes_animal_ingredients) > 0 else 'keine'))
    print('=' * 64)
    print('Gefundene Milchprodukte: ' + (', '.join(milk_ingredients) if len(milk_ingredients) > 0 else 'keine'))
    print('=' * 64)
    print('Gefundene Gluten: ' + (', '.join(gluten_ingredients) if len(gluten_ingredients) > 0 else 'keine'))
    print('=' * 64)
    print(LEGAL_NOTICE)
    print('=' * 64)


if __name__ == '__main__':
    main()
