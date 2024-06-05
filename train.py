import torch
import torchvision
import os
import pandas as pd
from PIL import Image


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 100
BATCH_SIZE = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = (72, 128)


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
        for file in sorted(os.listdir(relative_path('data/synthetic_frames'))):
            if file.split('.')[-1].strip().lower() == 'png':
                for data_file in sorted(os.listdir(relative_path('data/synthetic_frames_json'))):
                    if data_file.split('.')[0] == file.split('.')[0]:
                        self.images.append(file)
                        self.data.append(data_file)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(relative_path(f'data/synthetic_frames/{self.images[index]}'))
        width, height = image.size
        image.thumbnail(IMAGE_SIZE, Image.Resampling.LANCZOS)
        data = pd.read_json(relative_path(f'data/synthetic_frames_json/{self.data[index]}'))
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
        self.fc1 = torch.nn.Linear(16 * 15 * 29, 240)
        self.fc2 = torch.nn.Linear(240, 120)
        self.fc3 = torch.nn.Linear(120, 12)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 15 * 29)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    train_dataset = SyntheticDataset(train=True, transform=transform)
    test_dataset = SyntheticDataset(train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    model = ConvNet().to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    n_total_steps = len(train_loader)
    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, (images, data) in enumerate(train_loader):
            images = images.to(DEVICE)
            data = data.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 20 == 0:
                print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}")
    print('Finished Training')
    model.eval()
    with torch.no_grad():
        n_total_distance = 0.0
        n_total_count = 0
        for images, data in test_loader:
            n_total_count += len(images)
            images = images.to(DEVICE)
            data = data.to(DEVICE)
            outputs = model(images)
            distance = torch.abs(outputs - data)
            n_total_distance += torch.sum(distance).item()
        print(f"Average distance: {n_total_distance / (n_total_count * 12):.4f}")


if __name__ == '__main__':
    main()
