import torch
import torchvision
import os
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
    'decrease_size',
    'PROMPT_LLM',
    'PROMPT_CLAUDE',
    'PROMPT_VISION',
    'EOS',
    'GRAMMAR',
    'SYSTEM_PROMPT',
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
SYSTEM_PROMPT = 'Du bist ein hilfreicher assistant.'
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


def decrease_size(input_path, output_path, max_size, max_side):
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
