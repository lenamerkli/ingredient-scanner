from os.path import join, dirname
from os import listdir
from tqdm import tqdm

import ffmpeg  # noqa


VIDEO_EXTENSIONS = [
    'mov',
    'mp4',
]


def relative_path(string: str) -> str:
    return join(dirname(__file__), string)


def main():
    for file in tqdm(listdir(relative_path('videos'))):
        if file.rsplit('.', 1)[1].strip().lower() in VIDEO_EXTENSIONS:
            (
                ffmpeg
                .input(relative_path(f"videos/{file}"))
                .output(relative_path('frames/frame%05d.png'))
                .run()
            )


if __name__ == '__main__':
    main()
