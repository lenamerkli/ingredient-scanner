from general import (
    CRITERIONS,
    ConvNet,
    DEVICE,
    IMAGE_SIZE,
    Path,
    SyntheticDataset,
    current_time,
    json,
    relative_path,
    sqrt,
    torch,
    torchvision,
)

NUM_EPOCHS = 256
BATCH_SIZE = 16
LEARNING_RATE = 0.0004
CRITERION = 'SmoothL1Loss'


def required_loss(loss_history: list[float]) -> float:
    return sorted(loss_history[-(NUM_EPOCHS // 10):])[1]


def main():
    start_time = current_time()
    for path in ['tmp/frames', 'tmp/synthetic_frames']:
        Path(relative_path(path)).mkdir(parents=True, exist_ok=True)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    train_dataset = SyntheticDataset(train=True, transform=transform)
    test_dataset = SyntheticDataset(train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    model = ConvNet().to(DEVICE)
    criterion = CRITERIONS[CRITERION]()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    n_total_steps = len(train_loader)
    i = 0
    loss_history = []
    train = True
    epoch = 0
    while train:
        model.train()
        for i, (images, data) in enumerate(train_loader):
            images = images.to(DEVICE)
            data = data.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        epoch += 1
        if epoch >= NUM_EPOCHS and loss_history[-1] <= required_loss(loss_history):
            train = False
        if loss_history:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}], Step [{i + 1}/{n_total_steps}], Loss: {loss_history[-1]:.6f}")
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
            distance = torch.abs(outputs - data).tolist()
            for i in range(len(distance)):
                for j in range(len(distance[i]) // 2):
                    n_total_distance += sqrt(distance[i][2 * j] ** 2 + distance[i][2 * j + 1] ** 2)
    average_distance = n_total_distance / (n_total_count * 12)
    print(f"Average distance: {average_distance:.4f}")
    time_stamp = current_time()
    torch.save(model.state_dict(), relative_path(f"models/{time_stamp}.pt"))
    model_data = {
        'average_distance': average_distance,
        'batch_size': BATCH_SIZE,
        'criterion': CRITERION,
        'image_size_x': IMAGE_SIZE[0],
        'image_size_y': IMAGE_SIZE[1],
        'last_loss': loss_history[-1],
        'learning_rate': LEARNING_RATE,
        'loss_ao10': sum(loss_history[-10:-1]) / 10.0 if len(loss_history) >= 10 else None,
        'num_epochs': NUM_EPOCHS,
        'start_time': start_time,
        'type': 'base',
    }
    with open(relative_path(f"models/{time_stamp}.json"), 'w') as f:
        json.dump(model_data, f, indent=2)
    print(f"Model saved as {time_stamp}.pt")


if __name__ == '__main__':
    main()
