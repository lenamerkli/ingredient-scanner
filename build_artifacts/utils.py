import torch
import torchvision
import os
import json
from PIL import Image
from datetime import datetime


__all__ = [
    'ANIMAL',
    'DEVICE',
    'GLUTEN',
    'GRAMMAR',
    'IMAGE_SIZE',
    'LEGAL_NOTICE',
    'MARGIN',
    'MILK',
    'NeuralNet',
    'PROMPT_CLAUDE',
    'PROMPT_LLM',
    'PROMPT_VISION',
    'SOMETIMES_ANIMAL',
    'SYSTEM_PROMPT',
    'TRANSFORM',
    'decrease_size',
    'relative_path',
]


MARGIN: float = 0.1  # how much padding should be added to the cut-out image
# auto-select GPU if cuda is available
DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE: tuple[int, int] = (224, 224)  # input image size of the vision model
TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])  # transformations to the image before the pass to the vision model
with open('prompt_llm.md', 'r', encoding='utf-8') as _f:
    PROMPT_LLM: str = _f.read()  # prompt for the local LLM
with open('prompt_claude.md', 'r', encoding='utf-8') as _f:
    PROMPT_CLAUDE: str = _f.read()  # prompt for the Anthropic API
with open('prompt_vision.md', 'r', encoding='utf-8') as _f:
    PROMPT_VISION: str = _f.read()  # task for the local optical character recognition model
SYSTEM_PROMPT: str = 'Du bist ein hilfreicher assistant.'
with open('grammar.gbnf', 'r', encoding='utf-8') as _f:
    GRAMMAR: str = _f.read()
# ingredients list
with open('animal.json', 'r', encoding='utf-8') as _f:
    ANIMAL: list[str] = json.load(_f)
with open('sometimes_animal.json', 'r', encoding='utf-8') as _f:
    SOMETIMES_ANIMAL: list[str] = json.load(_f)
with open('milk.json', 'r', encoding='utf-8') as _f:
    MILK: list[str] = json.load(_f)
with open('gluten.json', 'r', encoding='utf-8') as _f:
    GLUTEN: list[str] = json.load(_f)


LEGAL_NOTICE: str = ('Dieses Programm ist nur für Forschungszwecke gedacht. Fehler können nicht ausgeschlossen werden '
                     'und sind wahrscheinlich vorhanden. Die Erkennung von Zutaten und Verunreinigungen ist nur zum '
                     'schnellen Aussortieren und nicht zum Überprüfen gedacht.')


def relative_path(string: str) -> str:
    """
    Returns the absolute path of a given string by joining it with the directory of the current file.
    :param string: relative path
    :return: absolute path
    """
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
        # Check the need to unsqueeze
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


def decrease_size(input_path: str, output_path: str, max_size: int, max_side: int):
    """
    Decrease the size of an image and convert it to WEBP.
    :param input_path: input path
    :param output_path: output path, preferably with .webp extension
    :param max_size: maximum file size in bytes
    :param max_side: maximum resolution in pixels
    :return: None
    """
    with Image.open(input_path) as img:
        original_size = os.path.getsize(input_path)
        width, height = img.size
        if original_size <= max_size and width <= max_side and height <= max_side:
            img.save(output_path, format=output_path.split('.')[-1].upper())
            print("Image is already below the maximum size.")
        while width > 24 and height > 24:
            img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
            img_resized.save(output_path, format=output_path.split('.')[-1].upper())
            if os.path.getsize(output_path) <= max_size and width <= max_side and height <= max_side:
                print(f"Reduced image size to {os.path.getsize(output_path)} bytes.")
                break
            width, height = int(width * 0.9), int(height * 0.9)
        if os.path.getsize(output_path) > max_size:
            raise ValueError("Could not reduce PNG size below max_size by reducing resolution.")
