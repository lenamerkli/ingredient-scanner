from general import (
    tqdm,
    json,
    torch,
    relative_path,
    os,
    NeuralNet,
    DEVICE,
    IMAGE_SIZE,
    Image,
    torchvision,
)


ORIGINAL_SIZE = (720, 1280)


def main() -> None:
    model_name = sorted([i for i in os.listdir(relative_path('models')) if i.endswith('.pt')])[-1]
    model_weights = torch.load(relative_path(f'models/{model_name}'))
    model = NeuralNet()
    model.load_state_dict(model_weights)
    model.to(DEVICE)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    model.eval()
    with torch.no_grad():
        for file in tqdm(os.listdir(relative_path('data/frames'))):
            if file.endswith('.png'):
                image = Image.open(relative_path(f"data/frames/{file}"))
                image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                image = transform(image).to(DEVICE)
                output = model(image).tolist()[0]
                data = {
                    'top': {
                        'left': {
                            'x': output[0] * ORIGINAL_SIZE[0],
                            'y': output[1] * ORIGINAL_SIZE[1],
                        },
                        'right': {
                            'x': output[2] * ORIGINAL_SIZE[0],
                            'y': output[3] * ORIGINAL_SIZE[1],
                        },
                    },
                    'bottom': {
                        'left': {
                            'x': output[4] * ORIGINAL_SIZE[0],
                            'y': output[5] * ORIGINAL_SIZE[1],
                        },
                        'right': {
                            'x': output[6] * ORIGINAL_SIZE[0],
                            'y': output[7] * ORIGINAL_SIZE[1],
                        },
                    },
                    'curvature': {
                        'top': {
                            'x': output[8] * ORIGINAL_SIZE[0],
                            'y': output[9] * ORIGINAL_SIZE[1],
                        },
                        'bottom': {
                            'x': output[10] * ORIGINAL_SIZE[0],
                            'y': output[11] * ORIGINAL_SIZE[1],
                        },
                    },
                }
                with open(relative_path(f"data/rhlf_json/{file.split('.')[0]}.json"), 'w') as f:
                    json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()
