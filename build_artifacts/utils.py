import torch
import torchvision
import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime


__all__ = [
    'current_time',
    'relative_path',
    'NeuralNet',
    'DEVICE',
    'IMAGE_SIZE',
    'TRANSFORM',
    'MARGIN',
    'GRID_SIZE',
    'apply_custom_transform',
    'generate_grid',
    'decrease_size',
    'PROMPT_LLM',
    'PROMPT_CLAUDE',
    'PROMPT_VISION',
    'EOS',
    'GRAMMAR',
]


MARGIN = 0.1
GRID_SIZE = 4096
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (224, 224)
TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
with open('prompt_llm.md', 'r', encoding='utf-8') as _f:
    PROMPT_LLM = _f.read()
with open('prompt_claude.md', 'r', encoding='utf-8') as _f:
    PROMPT_CLAUDE = _f.read()
with open('prompt_vision.md', 'r', encoding='utf-8') as _f:
    PROMPT_VISION = _f.read()
EOS = '\n<|im_end|>'
with open('grammar.gbnf', 'r', encoding='utf-8') as _f:
    GRAMMAR = _f.read()


def current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def relative_path(string: str) -> str:
    return os.path.join(os.path.dirname(__file__), string)


class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # Load pre-trained ResNet model
        self.backbone = torchvision.models.resnet18(pretrained=True)

        # Modify the last layer to output 12 values
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, 12)

        # Add a custom head for key-point detection
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 12, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        # Check if we need to unsqueeze
        if len(x.shape) == 3:  # Shape [C, H, W]
            x = x.unsqueeze(0)  # Shape [1, C, H, W]

        # Resize input to match ResNet input size if necessary
        if x.shape[-2:] != (224, 224):
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Pass input through the backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Pass input through the custom head
        x = self.head(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        return x


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


def decrease_size(input_path, output_path, max_size, max_side):
    with Image.open(input_path) as img:
        original_size = os.path.getsize(input_path)
        if original_size <= max_size:
            print("Image is already below the maximum size.")
            return True
        width, height = img.size
        while width > 24 and height > 24:
            img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
            img_resized.save(output_path, format=output_path.split('.')[-1].upper())
            if os.path.getsize(output_path) <= max_size and width <= max_side and height <= max_side:
                print(f"Reduced image size to {os.path.getsize(output_path)} bytes.")
                break
            width, height = int(width * 0.9), int(height * 0.9)
        if os.path.getsize(output_path) > max_size:
            raise ValueError("Could not reduce PNG size below max_size by reducing resolution.")
