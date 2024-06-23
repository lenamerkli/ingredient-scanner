import easyocr
import requests
import base64
from dotenv import load_dotenv

from general import relative_path, tqdm, os, json, Image

MAX_SIZE = 5_000_000
ENGINE = ['anthropic', 'claude-3-5-sonnet-20240620', 'claude_3_5_sonnet']
with open(relative_path('data2/prompt.md'), 'r', encoding='utf-8') as _f:
    PROMPT = _f.read()

load_dotenv()


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
        input_path = relative_path(f'data2/frames/{file}.png')
        output_path = relative_path(f'tmp/frames_claude/{file}.webp')
        with Image.open(input_path) as img:
            original_size = os.path.getsize(input_path)
            if original_size <= MAX_SIZE:
                print("Image is already below the maximum size.")
                return True

            # Start reducing the size by halving the resolution until it's below max_size
            width, height = img.size
            while width > 0 and height > 0:
                width, height = int(width * 0.9), int(height * 0.9)
                img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
                img_resized.save(output_path, format='WEBP')
                if os.path.getsize(output_path) <= MAX_SIZE:
                    print(f"Reduced image size to {os.path.getsize(output_path)} bytes.")
                    break
            if os.path.getsize(output_path) > MAX_SIZE:
                raise ValueError("Could not reduce PNG size below max_size by reducing resolution.")
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
        with open(relative_path(f"data2/frames_claude/{file}.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(data)


if __name__ == '__main__':
    main()
