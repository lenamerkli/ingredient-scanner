from os.path import join, dirname
from os import listdir
from tqdm import tqdm

import ffmpeg  # noqa


VIDEO_EXTENSIONS = [
    'mov',
    'mp4',
]


def relative_path(string: str) -> str:
    """
    Returns the absolute path of a given string by joining it with the directory of the current file.
    :param string:
    :return:
    """
    return join(dirname(__file__), string)


def main() -> None:
    """
    Converts video files in the 'videos' directory to individual frames and saves them as PNG images in the 'frames' directory.
    :return: None
    """
    for file in tqdm(listdir(relative_path('videos'))):
        if file.split('.')[-1].strip().lower() in VIDEO_EXTENSIONS:
            (
                # call ffmpeg to do the job
                ffmpeg
                .input(relative_path(f"videos/{file}"))
                .output(relative_path(f"frames/{file.rsplit('.', 1)[0]}_%04d.png"), loglevel='quiet')
                .run()
            )


if __name__ == '__main__':
    main()
