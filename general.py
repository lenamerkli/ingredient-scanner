from PIL import Image
from datetime import datetime
from pathlib import Path
from math import sqrt
from tqdm import tqdm
from torchviz import make_dot
import json
import os
import pandas as pd
import torch
import torchvision


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (224, 224)
ORIGINAL_SIZE = (720, 1280)
CRITERIONS = {
    'BCELoss': torch.nn.BCELoss,
    'CTCLoss': torch.nn.CTCLoss,
    'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
    'L1Loss': torch.nn.L1Loss,
    'MSELoss': torch.nn.MSELoss,
    'NLLLoss': torch.nn.NLLLoss,
    'SmoothL1Loss': torch.nn.SmoothL1Loss,
}


def current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def relative_path(string: str) -> str:
    return os.path.join(os.path.dirname(__file__), string)


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None):
        if train:
            self.image_dir = 'synthetic_frames'
            self.data_dir = 'synthetic_frames_json'
        else:
            self.image_dir = 'frames'
            self.data_dir = 'frames_json'
        self.images = []
        self.data = []
        self.transform = transform
        for file in sorted(os.listdir(relative_path(f"data/full_images/{self.image_dir}"))):
            if file.split('.')[-1].strip().lower() == 'png':
                for data_file in sorted(os.listdir(relative_path(f"data/{self.data_dir}"))):
                    if data_file.split('.')[0] == file.split('.')[0]:
                        image = Image.open(relative_path(f"data/full_images/{self.image_dir}/{file}"))
                        width, height = (720, 1280)
                        data = pd.read_json(relative_path(f"data/{self.data_dir}/{data_file}"))
                        if data['curvature']['top']['x'] is None:
                            data['curvature']['top']['x'] = (data['top']['left']['x'] + data['top']['right']['x']) / 2
                        if data['curvature']['top']['y'] is None:
                            data['curvature']['top']['y'] = (data['top']['left']['y'] + data['top']['right']['y']) / 2
                        if data['curvature']['bottom']['x'] is None:
                            data['curvature']['bottom']['x'] = (data['bottom']['left']['x'] + data['bottom']['right'][
                                'x']) / 2
                        if data['curvature']['bottom']['y'] is None:
                            data['curvature']['bottom']['y'] = (data['bottom']['left']['y'] + data['bottom']['right'][
                                'y']) / 2
                        if image.size != ORIGINAL_SIZE:
                            for key1 in data:
                                for key2 in data[key1]:
                                    data[key1][key2]['x'] = int(data[key1][key2]['x'] * (width / ORIGINAL_SIZE[0]))
                                    data[key1][key2]['y'] = int(data[key1][key2]['y'] * (height / ORIGINAL_SIZE[1]))
                            image = image.resize(ORIGINAL_SIZE, Image.Resampling.LANCZOS)
                        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                        image.save(relative_path(f"tmp/{self.image_dir}/{file}"))
                        tensor_data = [
                            data['top']['left']['x'] / width,
                            data['top']['left']['y'] / height,
                            data['top']['right']['x'] / width,
                            data['top']['right']['y'] / height,
                            data['bottom']['left']['x'] / width,
                            data['bottom']['left']['y'] / height,
                            data['bottom']['right']['x'] / width,
                            data['bottom']['right']['y'] / height,
                            data['curvature']['top']['x'] / width,
                            data['curvature']['top']['y'] / height,
                            data['curvature']['bottom']['x'] / width,
                            data['curvature']['bottom']['y'] / height,
                        ]
                        if any([0.0 > x or 1.0 < x for x in tensor_data]):
                            continue
                        tensor_data = torch.tensor(tensor_data, dtype=torch.float32).to(DEVICE)
                        if self.transform:
                            image = self.transform(image).to(DEVICE)
                        self.data.append(tensor_data)
                        self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        data = self.data[index]
        return image, data


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


if __name__ == '__main__':
    _model = NeuralNet()
    _x = torch.rand(1, 3, 224, 224)
    _dot = make_dot(_model(_x), params=dict(_model.named_parameters()))
    _dot.format = 'png'
    _dot.render('model_graph.png')
