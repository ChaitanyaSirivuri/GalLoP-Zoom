#!/bin/bash

# 1. Install GroundingDINO
echo "Cloning GroundingDINO..."
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/

echo "Installing dependencies..."
pip install -q -e .
pip install -q huggingface_hub datasets

# 2. Download Weights
echo "Downloading weights..."
mkdir -p weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
cd ..

echo "Setup Complete. You can now run 'python run_dino_colab.py'"
