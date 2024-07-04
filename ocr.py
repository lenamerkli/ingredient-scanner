import requests
import base64
import easyocr
from dotenv import load_dotenv
from general import relative_path, os, json, Image, tqdm

MAX_SIZE = 5_000_000
# ENGINE = ['easyocr']
# ENGINE = ['anthropic', 'claude-3-5-sonnet-20240620', 'claude_3_5_sonnet']
ENGINE = ['llama_cpp/v2/vision', 'qwen-vl-next_b2583']
with open(relative_path('data/cropped_images/prompt.md'), 'r', encoding='utf-8') as _f:
    PROMPT = _f.read()

load_dotenv()


def decrease_size(input_path, output_path):
    with Image.open(input_path) as img:
        original_size = os.path.getsize(input_path)
        if original_size <= MAX_SIZE:
            print("Image is already below the maximum size.")
            return True
        width, height = img.size
        while width > 0 and height > 0:
            img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
            img_resized.save(output_path, format=output_path.split('.')[-1].upper())
            if os.path.getsize(output_path) <= MAX_SIZE:
                print(f"Reduced image size to {os.path.getsize(output_path)} bytes.")
                break
            width, height = int(width * 0.9), int(height * 0.9)
        if os.path.getsize(output_path) > MAX_SIZE:
            raise ValueError("Could not reduce PNG size below max_size by reducing resolution.")


def main():
    if ENGINE[0] == 'easyocr':
        reader = easyocr.Reader(['de', 'fr', 'en'], gpu=True)
        for file in tqdm(os.listdir(relative_path('data2/frames'))):
            if file.endswith('.png'):
                result = reader.readtext(relative_path(f'data2/frames/{file}'))
                with open(relative_path(f"data2/frames_ocr/{file.rsplit('.', 1)[0]}.txt"), 'w', encoding='utf-8') as f:
                    f.write('\n'.join([i[1] for i in result]))
    elif ENGINE[0] == 'anthropic':
        file = input('Enter file name without extension: ')
        input_path = relative_path(f"data/cropped_images/frames/{file}.png")
        output_path = relative_path(f"tmp/frames_claude/{file}.webp")
        decrease_size(input_path, output_path)
        with open(output_path, 'rb') as f:
            image = base64.b64encode(f.read()).decode('utf-8')
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
                                    'data': image,
                                },
                            },
                            {
                                'type': 'text',
                                'text': PROMPT,
                            },
                        ],
                    },
                ],
            }),
        )
        data = response.json()
        with open(relative_path(f"data/cropped_images/frames_claude/{file}.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(data)
    elif ENGINE[0] == 'llama_cpp/v2/vision':
        for file in tqdm(os.listdir(relative_path('data/cropped_images/frames'))):
            if file.endswith('.png'):
                input_path = relative_path(f"data/cropped_images/frames/{file}")
                output_path = relative_path(f"tmp/frames_local/{file.rsplit('.', 1)[0]}.webp")
                decrease_size(input_path, output_path)
                response = requests.post(
                    url='http://127.0.0.1:11434/llama_cpp/v2/vision',
                    headers={
                        'x-version': '2024-05-21',
                        'content-type': 'application/json',
                    },
                    data=json.dumps({
                        'task': PROMPT,
                        'model': ENGINE[1],
                        'image_path': output_path,
                    }),
                )
                data: str = response.json()['text']
                with open(relative_path(f"data/cropped_images/frames_local/{file.rsplit('.', 1)[0]}.txt"),
                          'w', encoding='utf-8') as f:
                    f.write(data)
                print(data)


if __name__ == '__main__':
    main()
