import cv2
import numpy as np
from general import relative_path, json


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

    # Interpolate the top and bottom curves
    top_curve_x = np.linspace(tl[0], tr[0], nx)
    top_curve_y = np.linspace(tl[1], tr[1], nx)
    bottom_curve_x = np.linspace(bl[0], br[0], nx)
    bottom_curve_y = np.linspace(bl[1], br[1], nx)

    curvature_top_x = np.linspace(tl[0], tr[0], nx)
    curvature_top_y = np.linspace(tl[1], tr[1], nx) + (tm[1] - np.mean([tl[1], tr[1]])) * np.sin(
        np.linspace(0, np.pi, nx))

    curvature_bottom_x = np.linspace(bl[0], br[0], nx)
    curvature_bottom_y = np.linspace(bl[1], br[1], nx) + (bm[1] - np.mean([bl[1], br[1]])) * np.sin(
        np.linspace(0, np.pi, nx))

    for i in range(ny):
        t = i / (ny - 1)
        inter_curv_top_x = (1 - t) * top_curve_x + t * curvature_top_x
        inter_curv_top_y = (1 - t) * top_curve_y + t * curvature_top_y
        inter_curv_bottom_x = (1 - t) * curvature_bottom_x + t * bottom_curve_x
        inter_curv_bottom_y = (1 - t) * curvature_bottom_y + t * bottom_curve_y

        grid_x[i, :] = (1 - t) * inter_curv_top_x + t * inter_curv_bottom_x
        grid_y[i, :] = (1 - t) * inter_curv_top_y + t * inter_curv_bottom_y

    return grid_x, grid_y


def apply_custom_transform(image, grid_x, grid_y):
    """
    Apply a custom transformation defined by grid mapping.
    :param image: Input image.
    :param grid_x: X coordinates of the destination grid.
    :param grid_y: Y coordinates of the destination grid.
    :return: Warped image with the custom transform applied.
    """
    warped = cv2.remap(image, grid_x, grid_y, cv2.INTER_CUBIC)
    return warped


def main():
    # load the image and grab the points from the user
    image = cv2.imread(relative_path('data/frames/07432_0007.png'))
    with open(relative_path('data/frames_json/07432_0007.json'), 'r') as json_file:
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
        (data['top']['left']['x'], data['top']['left']['y']),
        (data['top']['right']['x'], data['top']['right']['y']),
        (data['bottom']['left']['x'], data['bottom']['left']['y']),
        (data['bottom']['right']['x'], data['bottom']['right']['y']),
        (data['curvature']['top']['x'], data['curvature']['top']['y']),
        (data['curvature']['bottom']['x'], data['curvature']['bottom']['y']),
    ])

    # Grid size aka. number of divisions in the grid
    nx, ny = 1024, 1024

    # Generate the grid
    grid_x, grid_y = generate_grid(nx, ny, points)

    # Apply the custom transformation
    warped_image = apply_custom_transform(image, grid_x, grid_y)

    if warped_image is not None:
        # show the original and warped images
        cv2.imshow("Warped", warped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to warp the image.")


if __name__ == '__main__':
    main()
