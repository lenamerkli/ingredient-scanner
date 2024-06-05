from PIL import Image
from datetime import datetime
from pathlib import Path
import json
import os
import pandas as pd
import torch
import torchvision


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (72 * 3, 128 * 3)


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
        for file in sorted(os.listdir(relative_path(f"data/{self.image_dir}"))):
            if file.split('.')[-1].strip().lower() == 'png':
                for data_file in sorted(os.listdir(relative_path(f"data/{self.data_dir}"))):
                    if data_file.split('.')[0] == file.split('.')[0]:
                        image = Image.open(relative_path(f"data/{self.image_dir}/{file}"))
                        image.thumbnail(IMAGE_SIZE, Image.Resampling.LANCZOS)
                        image.save(relative_path(f"tmp/{self.image_dir}/{file}"))
                        self.images.append(file)
                        self.data.append(data_file)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(relative_path(f"tmp/{self.image_dir}/{self.images[index]}"))
        width, height = (720, 1280)
        data = pd.read_json(relative_path(f"data/{self.data_dir}/{self.data[index]}"))
        if data['curvature']['top']['x'] is None:
            data['curvature']['top']['x'] = (data['top']['left']['x'] + data['top']['right']['x']) / 2
        if data['curvature']['top']['y'] is None:
            data['curvature']['top']['y'] = (data['top']['left']['y'] + data['top']['right']['y']) / 2
        if data['curvature']['bottom']['x'] is None:
            data['curvature']['bottom']['x'] = (data['bottom']['left']['x'] + data['bottom']['right']['x']) / 2
        if data['curvature']['bottom']['y'] is None:
            data['curvature']['bottom']['y'] = (data['bottom']['left']['y'] + data['bottom']['right']['y']) / 2
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
        tensor_data = torch.tensor(tensor_data, dtype=torch.float32).to(DEVICE)
        if self.transform:
            image = self.transform(image).to(DEVICE)
        return image, tensor_data


class ConvNet(torch.nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * (IMAGE_SIZE[0] // 4 - 3) * (IMAGE_SIZE[1] // 4 - 3), 256)
        self.fc2 = torch.nn.Linear(256, 96)
        self.fc3 = torch.nn.Linear(96, 12)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * (IMAGE_SIZE[0] // 4 - 3) * (IMAGE_SIZE[1] // 4 - 3))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
