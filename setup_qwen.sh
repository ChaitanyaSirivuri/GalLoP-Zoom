#!/bin/bash

# 1. Install Dependencies
echo "Installing Qwen2.5-VL Dependencies..."
# Standard PyTorch/HuggingFace libraries only. No custom C++ compilation required.
pip install -q git+https://github.com/huggingface/transformers accelerate
pip install -q qwen-vl-utils

# 2. Flash Attention (Optional but recommended for speed, usually pre-installed or easy)
# We limit to what's safe. standard attention works fine, just slower.
# pip install -q flash-attn --no-build-isolation

echo "Setup Complete. You can now run 'python run_qwen.py'"
