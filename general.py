from PIL import Image
from datetime import datetime
from pathlib import Path
from math import sqrt
from tqdm import tqdm
from zlib import adler32
from copy import deepcopy
from torchviz import make_dot
import json
import os
import pandas as pd
import torch
import torchvision


class IngredientScannerLoss(torch.nn.Module):
    def __init__(self, alpha=2.0, beta=1.2, **_):
        super(IngredientScannerLoss, self).__init__()
        self._alpha = alpha
        self._beta = beta
        
    def _apply_alpha_beta(self, distance):
        return ((distance + 1) ** self._beta - 1) * self._alpha

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        delta = output - target
        batched = len(delta.shape) > 1
        losses = torch.zeros(delta.shape[0], device=output.device)
        if not batched:
            delta = delta.unsqueeze(0)
        for i in range(delta.shape[0]):
            loss = 0.0
            for j in range(6):
                distance = torch.sqrt(delta[i][j * 2] ** 2 + delta[i][j * 2 + 1] ** 2)
                if j == 0:  # top left
                    if delta[i][0] > 0.0:
                        distance = self._apply_alpha_beta(distance)
                    if delta[i][1] > 0.0:
                        distance = self._apply_alpha_beta(distance)
                elif j == 1:  # top right
                    if delta[i][2] > 0.0:
                        distance = self._apply_alpha_beta(distance)
                    if delta[i][3] < 0.0:
                        distance = self._apply_alpha_beta(distance)
                elif j == 2:  # bottom left
                    if delta[i][4] < 0.0:
                        distance = self._apply_alpha_beta(distance)
                    if delta[i][5] < 0.0:
                        distance = self._apply_alpha_beta(distance)
                elif j == 3:  # bottom right
                    if delta[i][6] < 0.0:
                        distance = self._apply_alpha_beta(distance)
                    if delta[i][7] > 0.0:
                        distance = self._apply_alpha_beta(distance)
                elif j == 4:  # curvature top
                    if delta[i][9] > 0.0:
                        distance = self._apply_alpha_beta(distance)
                elif j == 5:  # curvature bottom
                    if delta[i][11] < 0.0:
                        distance = self._apply_alpha_beta(distance)
                loss += distance
            losses[i] = loss
        if not batched:
            return losses[0]
        return losses


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (224, 224)
ORIGINAL_SIZE = (720, 1280)
CRITERIONS = {
    'BCELoss': torch.nn.BCELoss,
    'CTCLoss': torch.nn.CTCLoss,
    'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
    'HingeEmbeddingLoss': torch.nn.HingeEmbeddingLoss,
    'IngredientScannerLoss': IngredientScannerLoss,
    'KLDivLoss': torch.nn.KLDivLoss,
    'L1Loss': torch.nn.L1Loss,
    'MarginRankingLoss': torch.nn.MarginRankingLoss,
    'MSELoss': torch.nn.MSELoss,
    'NLLLoss': torch.nn.NLLLoss,
    'SmoothL1Loss': torch.nn.SmoothL1Loss,
    'SoftMarginLoss': torch.nn.SoftMarginLoss,
    'TripletMarginLoss': torch.nn.TripletMarginLoss,
}


def current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def relative_path(string: str) -> str:
    return os.path.join(os.path.dirname(__file__), string)


def calculate_adler32_file_chunked(filepath, chunk_size=1024*64):
    checksum = 1  # Adler-32 initial value
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(chunk_size)  # Read the file in chunks
            if not data:
                break  # If there is no more data, exit the loop
            checksum = adler32(data, checksum)  # Calculate checksum of chunk
    return checksum


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
        if not os.path.exists(relative_path('tmp/hashes.json')):
            with open(relative_path('tmp/hashes.json'), 'w') as f:
                json.dump({}, f, indent=4)
        with open(relative_path('tmp/hashes.json'), 'r') as f:
            hashes: dict[str, int] = json.load(f)
        new_hashes = deepcopy(hashes)
        for file in tqdm(sorted(os.listdir(relative_path(f"data/full_images/{self.image_dir}")))):
            if file.split('.')[-1].strip().lower() == 'png':
                for data_file in sorted(os.listdir(relative_path(f"data/full_images/{self.data_dir}"))):
                    if data_file.split('.')[0] == file.split('.')[0]:
                        hashed = (calculate_adler32_file_chunked(relative_path(
                            f"data/full_images/{self.image_dir}/{file}"))
                                  + calculate_adler32_file_chunked(relative_path(
                                    f"data/full_images/{self.data_dir}/{data_file}")))
                        use_cache = False
                        if file.split('.')[0] in hashes:
                            if hashed == hashes[file.split('.')[0]]:
                                use_cache = True
                        new_hashes[file.split('.')[0]] = hashed
                        if use_cache:
                            image = Image.open(relative_path(f"tmp/{self.image_dir}/{file}"))
                            width, height = (720, 1280)
                            with open(relative_path(f"tmp/{self.data_dir}/{data_file}"), 'r') as f:
                                data: dict[str, dict[str, dict[str, int | float | None]]] = json.load(f)
                        else:
                            image = Image.open(relative_path(f"data/full_images/{self.image_dir}/{file}"))
                            width, height = (720, 1280)
                            with open(relative_path(f"data/full_images/{self.data_dir}/{data_file}"), 'r') as f:
                                data: dict[str, dict[str, dict[str, int | float | None]]] = json.load(f)
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
                            with open(relative_path(f"tmp/{self.data_dir}/{data_file}"), 'w') as f:
                                json.dump(data, f, indent=4)
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
        with open(relative_path(f"tmp/hashes.json"), 'w') as f:
            json.dump(new_hashes, f, indent=4)

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
