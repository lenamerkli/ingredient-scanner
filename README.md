# Ingredient Scanner
## Abstract
With the recent advancements in computer vision and optical character recognition and using a convolutional neural network to cut out the product from a picture, it has now become possible to reliably extract ingredient lists from the back of a product using the Anthropic API. Open-weight or even only on-device optical character recognition lacks the quality to be used in a production environment, although the progress in development is promising. The Anthropic API is also currently not feasible due to the high cost of 1 Swiss Franc per 100 pictures.

An inference example can be found on [HuggingFace](https://huggingface.co/lenamerkli/ingredient-scanner).

This is an entry for the [2024 Swiss AI competition](https://www.ki-wettbewerb.ch/).

## Table of Contents

0. [Abstract](#abstract)
1. [Installation](#installation)
2. [Usage](#usage)
3. [Inference](#inference)
4. [Citation](#citation)

## Report
Read the full report [here](https://huggingface.co/lenamerkli/ingredient-scanner/blob/main/ingredient-scanner.pdf).

## Installation
GNU/Linux is required. Debian GNU/Linux 12 or later is recommended.

### Git

#### Installation
```bash
sudo apt install git
```

#### Cloning
```bash
git clone https://github.com/lenamerkli/ingredient-scanner
cd ingredient-scanner
```

### GNU C Compiler
```bash
sudo apt install gcc
```

### FFmpeg
```bash
sudo apt install ffmpeg
```

### NVIDIA driver
Install the proprietary driver from [NVIDIA](https://www.nvidia.com/en-us/drivers/unix/).

Make sure that all PATH variables are set correctly.

### CUDA
Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=12&target_type=deb_network), version 12.1.0 or later. Tested with version 12.5.0. Additional information can be found on the [NVIDIA website](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

Make sure that all PATH variables are set correctly.

### Python
Install Python 3.11.9 or later, below version 3.12:

```bash
sudo apt update
sudo apt install python3.11
```

#### Virtual environment
```bash
sudo apt install python3.11-venv
python3 -m venv .venv
source .venv/bin/activate
```

#### Libraries
```bash
pip3 install nvidia-pyindex
pip3 install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"
```

### tkinter
```bash
sudo apt install python3-tk
```

### Convert videos to frames
```bash
cd data/full_images
python3 video_to_frames.py
```

### Optional: Anthropic API
Create a file named `.env` with the following content:
```
OPENAI_API_KEY=YOUR_API_KEY
```
Replace `YOUR_API_KEY` with your API key which you can find in your [anthropic console](https://console.anthropic.com/settings/keys).

## Usage

### Generate synthetic images
```bash
cd data/full_images
python3 generate_synthetic.py
```

### Train the model
```bash
python3 train.py
```

### Generate synthetic text
```bash
cd data/ingredients
python3 generate_synthetic.py
```

### Train the large language model
```bash
python3 train_llm.py
```

### Build the inference project
```bash
python3 build.py
```

## Inference
A working example with trained models can be found on [HuggingFace](https://huggingface.co/lenamerkli/ingredient-scanner).

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
