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


def main() -> None:
    """
    Run the model over all images.
    :return: None
    """
    # select the most recent model
    model_name = sorted([i for i in os.listdir(relative_path('models')) if i.endswith('.pt')])[-1]
    # load the model
    model_weights = torch.load(relative_path(f'models/{model_name}'))
    model = NeuralNet()
    model.load_state_dict(model_weights)
    model.to(DEVICE)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
        for file in tqdm(os.listdir(relative_path('data/full_images/frames'))):
            if file.endswith('.png'):
                image = Image.open(relative_path(f"data/full_images/frames/{file}"))
                image_size = image.size
                image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                image = transform(image).to(DEVICE)
                output = model(image).tolist()[0]
                data = {
                    'top': {
                        'left': {
                            'x': output[0] * image_size[0],
                            'y': output[1] * image_size[1],
                        },
                        'right': {
                            'x': output[2] * image_size[0],
                            'y': output[3] * image_size[1],
                        },
                    },
                    'bottom': {
                        'left': {
                            'x': output[4] * image_size[0],
                            'y': output[5] * image_size[1],
                        },
                        'right': {
                            'x': output[6] * image_size[0],
                            'y': output[7] * image_size[1],
                        },
                    },
                    'curvature': {
                        'top': {
                            'x': output[8] * image_size[0],
                            'y': output[9] * image_size[1],
                        },
                        'bottom': {
                            'x': output[10] * image_size[0],
                            'y': output[11] * image_size[1],
                        },
                    },
                }
                with open(relative_path(f"data/full_images/rhlf_json/{file.split('.')[0]}.json"), 'w') as f:
                    json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()
