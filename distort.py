import cv2
import numpy as np
from general import relative_path, json, tqdm, os, ORIGINAL_SIZE


MARGIN: float = 0.1  # how much padding should be added to the cut-out image
SCALE_FACTOR: int = 4  # how much the size of the cut-out image should be scaled up


def main() -> None:
    """
    Warps all frames which have corresponding point data.
    :return: None
    """
    for file in tqdm(os.listdir(relative_path('data/full_images/frames'))):
        if (file.endswith('.png') and
                os.path.exists(relative_path(f'data/full_images/frames_json/{file.rsplit(".", 1)[0]}.json'))):
            image = cv2.imread(relative_path(f"data/full_images/frames/{file}"))
            with open(relative_path(f"data/full_images/frames_json/{file.rsplit('.', 1)[0]}.json"), 'r') as json_file:
                data = json.load(json_file)
            if data['curvature']['top']['x'] is None:
                data['curvature']['top']['x'] = (data['top']['left']['x'] + data['top']['right']['x']) / 2
            if data['curvature']['top']['y'] is None:
                data['curvature']['top']['y'] = (data['top']['left']['y'] + data['top']['right']['y']) / 2
            if data['curvature']['bottom']['x'] is None:
                data['curvature']['bottom']['x'] = (data['bottom']['left']['x'] + data['bottom']['right']['x']) / 2
            if data['curvature']['bottom']['y'] is None:
                data['curvature']['bottom']['y'] = (data['bottom']['left']['y'] + data['bottom']['right']['y']) / 2
            if image.shape[0] != ORIGINAL_SIZE[1] or image.shape[1] != ORIGINAL_SIZE[0]:
                for key1 in data:
                    for key2 in data[key1]:
                        data[key1][key2]['x'] = int(data[key1][key2]['x'] * (ORIGINAL_SIZE[0] / image.shape[1]))
                        data[key1][key2]['y'] = int(data[key1][key2]['y'] * (ORIGINAL_SIZE[1] / image.shape[0]))
                image = cv2.resize(image, (ORIGINAL_SIZE[0], ORIGINAL_SIZE[1]))
            size_x = ((data['top']['right']['x'] - data['top']['left']['x']) +
                      (data['bottom']['right']['x'] - data['bottom']['left']['x'])) / 2
            size_y = ((data['top']['right']['y'] - data['top']['left']['y']) +
                      (data['bottom']['right']['y'] - data['bottom']['left']['y'])) / 2
            margin_x = size_x * MARGIN
            margin_y = size_y * MARGIN
            image_size = image.shape[:2]
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
            max_height = int(
                max([  # y: top left - bottom left, top right - bottom right, curvature top - curvature bottom
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
            homography, _ = cv2.findHomography(points, destination_points)
            warped_image = cv2.warpPerspective(image, homography, (max_width, max_height))

            if warped_image is not None:
                cv2.imwrite(relative_path(f"data/cropped_images/frames/{file}"), warped_image)
            else:
                print(f"Could not apply transformation to {file}")


if __name__ == '__main__':
    main()
