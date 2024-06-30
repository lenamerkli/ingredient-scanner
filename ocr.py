import requests
import base64
from dotenv import load_dotenv
from general import relative_path, os, json, Image

MAX_SIZE = 5_000_000
# ENGINE = ['anthropic', 'claude-3-5-sonnet-20240620', 'claude_3_5_sonnet']
ENGINE = ['local', 'llama_cpp/v2/vision', 'qwen-vl-next_b2583']
with open(relative_path('data2/prompt.md'), 'r', encoding='utf-8') as _f:
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
    if ENGINE[0] == 'anthropic':
        file = input('Enter file name without extension: ')
        input_path = relative_path(f'data2/frames/{file}.png')
        output_path = relative_path(f'tmp/frames_claude/{file}.webp')
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
        with open(relative_path(f"data2/frames_claude/{file}.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(data)
    elif ENGINE[0] == 'local':
        file = input('Enter file name without extension: ')
        input_path = relative_path(f'data2/frames/{file}.png')
        output_path = relative_path(f'tmp/frames_local/{file}.webp')
        decrease_size(input_path, output_path)
        response = requests.post(
            url=f"http://127.0.0.1:11434/{ENGINE[1]}",
            headers={
                'x-version': '2024-05-21',
                'content-type': 'application/json',
            },
            data=json.dumps({
                'task': PROMPT,
                'model': ENGINE[2],
                'image_path': output_path,
            }),
        )
        data: str = response.json()['text']
        with open(relative_path(f"data2/frames_local/{file}.txt"), 'w', encoding='utf-8') as f:
            f.write(data)
        print(data + '\n\n')
        data = '{' + data.split('{', 1)[-1]
        data = data.rsplit('}', 1)[0] + '}'
        data = data.replace('\\n', '\n')
        data = json.loads(data)
        with open(relative_path(f"data2/frames_local/{file}.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(data)


if __name__ == '__main__':
    main()
