import cv2
import numpy as np
from general import relative_path, json, tqdm, os, ORIGINAL_SIZE, sqrt


MARGIN = 0.1
GRID_SIZE = 4096


def generate_grid(nx, ny, corners):
    """
    Generate a grid for remapping based on corner points and curvature indicators.
    :param nx: Number of divisions along the width.
    :param ny: Number of divisions along the height.
    :param corners: Array of 6 points (tl, tr, bl, br, tm, bm).
    :return: Mapping coordinates for remapping.
    """
    tl, tr, bl, br, tm, bm = corners
    grid_x = np.zeros((ny, nx), dtype=np.float32)
    grid_y = np.zeros((ny, nx), dtype=np.float32)

    # Create linear interpolations for the sides
    left_x = np.linspace(tl[0], bl[0], ny)
    left_y = np.linspace(tl[1], bl[1], ny)
    right_x = np.linspace(tr[0], br[0], ny)
    right_y = np.linspace(tr[1], br[1], ny)

    # Adjust the top and bottom sides to include curvature
    top_x = np.linspace(tl[0], tr[0], nx)
    top_y = np.linspace(tl[1], tr[1], nx)
    bottom_x = np.linspace(bl[0], br[0], nx)
    bottom_y = np.linspace(bl[1], br[1], nx)

    # Curvature adjustments
    # Adjust by sine curve interpolation between middle and edge curvature points
    top_curve_adjust = ((tm[1] - np.mean([tl[1], tr[1]])) *
                        np.sin(np.linspace(0, np.pi, nx)))
    bottom_curve_adjust = ((bm[1] - np.mean([bl[1], br[1]])) *
                           np.sin(np.linspace(0, np.pi, nx)))

    top_y += top_curve_adjust
    bottom_y += bottom_curve_adjust

    for i in range(ny):
        inter_top_x = np.linspace(top_x[i], bottom_x[i], nx)
        inter_top_y = np.linspace(top_y[i], bottom_y[i], nx)

        inter_bottom_x = np.linspace(top_x[i], bottom_x[i], nx)
        inter_bottom_y = np.linspace(top_y[i], bottom_y[i], nx)

        t = i / (ny - 1)
        grid_x[i, :] = (1 - t) * inter_top_x + t * inter_bottom_x
        grid_y[i, :] = (1 - t) * inter_top_y + t * inter_bottom_y

    return grid_x, grid_y


def apply_custom_transform(image, grid_x, grid_y, aspect_ratio):
    """
    Apply a custom transformation defined by grid mapping.
    :param aspect_ratio: Aspect ratio of the output image.
    :param image: Input image.
    :param grid_x: X coordinates of the destination grid.
    :param grid_y: Y coordinates of the destination grid.
    :return: Warped image with the custom transform applied.
    """
    warped = cv2.remap(image, grid_x, grid_y, cv2.INTER_LANCZOS4)
    # rotate 90 degrees to the right due to the remapping
    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    # mirror the image due to the remapping
    warped = cv2.flip(warped, 1)
    # adjust for the aspect ratio
    warped = cv2.resize(warped, (int(aspect_ratio[0] * GRID_SIZE), int(aspect_ratio[1] * GRID_SIZE)),
                        interpolation=cv2.INTER_LANCZOS4)
    return warped


def main():
    for file in tqdm(os.listdir(relative_path('data/frames'))):
        if file.endswith('.png') and os.path.exists(relative_path(f'data/frames_json/{file.rsplit(".", 1)[0]}.json')):
            image = cv2.imread(relative_path(f"data/frames/{file}"))
            with open(relative_path(f"data/frames_json/{file.rsplit('.', 1)[0]}.json"), 'r') as json_file:
                data = json.load(json_file)
            if data['curvature']['top']['x'] is None:
                data['curvature']['top']['x'] = (data['top']['left']['x'] + data['top']['right']['x']) / 2
            if data['curvature']['top']['y'] is None:
                data['curvature']['top']['y'] = (data['top']['left']['y'] + data['top']['right']['y']) / 2
            if data['curvature']['bottom']['x'] is None:
                data['curvature']['bottom']['x'] = (data['bottom']['left']['x'] + data['bottom']['right']['x']) / 2
            if data['curvature']['bottom']['y'] is None:
                data['curvature']['bottom']['y'] = (data['bottom']['left']['y'] + data['bottom']['right']['y']) / 2
            points = np.array([
                (data['top']['left']['x'] * (1 - MARGIN),
                 data['top']['left']['y'] * (1 - MARGIN)),
                (min(data['top']['right']['x'] * (1 + MARGIN), ORIGINAL_SIZE[0] - 1),
                 data['top']['right']['y'] * (1 - MARGIN)),
                (data['bottom']['left']['x'] * (1 - MARGIN),
                 min(data['bottom']['left']['y'] * (1 + MARGIN), ORIGINAL_SIZE[1] - 1)),
                (min(data['bottom']['right']['x'] * (1 + MARGIN), ORIGINAL_SIZE[0] - 1),
                 min(data['bottom']['right']['y'] * (1 + MARGIN), ORIGINAL_SIZE[1] - 1)),
                (data['curvature']['top']['x'],
                 data['curvature']['top']['y'] * (1 - MARGIN)),
                (data['curvature']['bottom']['x'],
                 min(data['curvature']['bottom']['y'] * (1 + MARGIN), ORIGINAL_SIZE[1] - 1)),
            ])

            # Grid size aka. number of divisions in the grid
            nx, ny = GRID_SIZE, GRID_SIZE

            # Generate the grid
            grid_x, grid_y = generate_grid(nx, ny, points)

            # Calculate new aspect ratio
            aspect_ratio = (
                (sqrt((data['top']['right']['x'] - data['top']['left']['x']) ** 2 +
                      (data['top']['right']['y'] - data['top']['left']['y']) ** 2) +
                 sqrt((data['bottom']['right']['x'] - data['bottom']['left']['x']) ** 2 +
                      (data['bottom']['right']['y'] - data['bottom']['left']['y']) ** 2)) / 2,
                (sqrt((data['top']['left']['x'] - data['bottom']['left']['x']) ** 2 +
                      (data['top']['left']['y'] - data['bottom']['left']['y']) ** 2) +
                 sqrt((data['top']['right']['x'] - data['bottom']['right']['x']) ** 2 +
                      (data['top']['right']['y'] - data['bottom']['right']['y']) ** 2)) / 2
            )
            aspect_ratio = (1, aspect_ratio[1] / aspect_ratio[0])\
                if aspect_ratio[0] <= aspect_ratio[1]\
                else (aspect_ratio[0] / aspect_ratio[1], 1)

            # Apply the custom transformation
            warped_image = apply_custom_transform(image, grid_x, grid_y, aspect_ratio)

            if warped_image is not None:
                cv2.imwrite(relative_path(f"data2/frames/{file}"), warped_image)
            else:
                print(f"Could not apply transformation to {file}")


if __name__ == '__main__':
    main()
