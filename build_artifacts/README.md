---
language:
- de
pipeline_tag: image-to-text
---
# Ingredient Scanner
## Abstract

With the recent advancements in computer vision and optical character recognition and using a convolutional neural network to cut out the product from a picture, it has now become possible to reliably extract ingredient lists from the back of a product using the Anthropic API. Open-weight or even only on-device optical character recognition lacks the quality to be used in a production environment, although the progress in development is promising. The Anthropic API is also currently not feasible due to the high cost of 1 Swiss Franc per 100 pictures.

The training code and data is available on [GitHub](https://github.com/lenamerkli/ingredient-scanner/). This repository just contains an inference example and the [report](https://huggingface.co/lenamerkli/ingredient-scanner/blob/main/ingredient-scanner.pdf).

This is an entry for the [2024 Swiss AI competition](https://www.ki-wettbewerb.ch/).

## Table of Contents

0. [Abstract](#abstract)
1. [Report](#report)
2. [Model Details](#model-details)
3. [Usage](#usage)
4. [Citation](#citation)

## Report
Read the full report [here](https://huggingface.co/lenamerkli/ingredient-scanner/blob/main/ingredient-scanner.pdf).

## Model Details
This repository consists of two models, one vision model and a large language model.

### Vision Model
Custom convolutional neural network based on [ResNet18](https://pytorch.org/hub/pytorch_vision_resnet/). It detects the four corner points and the upper and lower limits of a product.

### Language Model
Converts the text from the optical character recognition engine which lies in-between the two models to JSON. It is fine-tuned from [unsloth/Qwen2-0.5B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Qwen2-0.5B-Instruct-bnb-4bit).

## Usage
Clone the repository and install the dependencies on any debian-based system:
```bash
git clone https://huggingface.co/lenamerkli/ingredient-scanner
cd ingredient-scanner
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```
Note: not all requirements are needed for inference, as both training and inference requirements are listed.

Select the OCR engine in `main.py` by uncommenting one of the lines 20 to 22:
```python
# ENGINE: list[str] = ['easyocr']
# ENGINE: list[str] = ['anthropic', 'claude-3-5-sonnet-20240620']
# ENGINE: list[str] = ['llama_cpp/v2/vision', 'qwen-vl-next_b2583']
```
Note: Qwen-VL-Next is not an official qwen model. This is only to protect business secrets of a private model.

Run the inference script:
```bash
python3 main.py
```

You will be asked to enter the file path to a PNG image.

### Anthropic API

If you want to use the Anthropic API, create a `.env` file with the following content:
```
ANTHROPIC_API_KEY=YOUR_API_KEY
```

## Citation
Here is how to cite this paper in the bibtex format:
```bibtex
@misc{merkli2024ingriedient-scanner,
    title={Ingredient Scanner: Automating Reading of Ingredient Labels with Computer Vision},
    author={Lena Merkli and Sonja Merkli},
    date={2024-07-16},
    url={https://huggingface.co/lenamerkli/ingredient-scanner},
}
```
