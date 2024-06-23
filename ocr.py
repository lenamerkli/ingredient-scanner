import easyocr

from general import relative_path, tqdm, os


def main():
    reader = easyocr.Reader(['de', 'fr', 'en'], gpu=True)
    for file in tqdm(os.listdir(relative_path('data2/frames'))):
        if file.endswith('.png'):
            result = reader.readtext(relative_path(f'data2/frames/{file}'))
            with open(relative_path(f"data2/frames_ocr/{file.rsplit('.', 1)[0]}.txt"), 'w') as f:
                f.write('\n'.join([i[1] for i in result]))


if __name__ == '__main__':
    main()
