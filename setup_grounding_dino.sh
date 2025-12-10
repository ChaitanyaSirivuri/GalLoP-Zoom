#!/bin/bash

# Setup script for GroundingDINO with CUDA configuration

# ============================================
# CUDA Configuration
# ============================================

# Check if CUDA is available
if command -v nvcc &> /dev/null; then
    NVCC_PATH=$(which nvcc)
    CUDA_PATH=$(dirname $(dirname $NVCC_PATH))
    export CUDA_HOME=$CUDA_PATH
    echo "CUDA found at: $CUDA_HOME"
    echo "CUDA version: $(nvcc --version | grep release)"
else
    echo "Warning: nvcc not found. CUDA may not be installed or not in PATH."
    echo "GroundingDINO will be compiled in CPU-only mode."
    
    # Try common CUDA paths
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
        echo "Found CUDA at /usr/local/cuda"
    elif [ -d "/usr/local/cuda-11.3" ]; then
        export CUDA_HOME=/usr/local/cuda-11.3
        echo "Found CUDA at /usr/local/cuda-11.3"
    elif [ -d "/usr/local/cuda-12.0" ]; then
        export CUDA_HOME=/usr/local/cuda-12.0
        echo "Found CUDA at /usr/local/cuda-12.0"
    fi
fi

# Add CUDA to PATH if CUDA_HOME is set
if [ -n "$CUDA_HOME" ]; then
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo "CUDA_HOME set to: $CUDA_HOME"
fi

# ============================================
# Clone and Install GroundingDINO
# ============================================

echo ""
echo "============================================"
echo "Installing GroundingDINO..."
echo "============================================"

# Clone repository if not exists
if [ ! -d "GroundingDINO" ]; then
    echo "Cloning GroundingDINO repository..."
    git clone https://github.com/IDEA-Research/GroundingDINO.git
else
    echo "GroundingDINO directory already exists, skipping clone."
fi

# Install GroundingDINO
cd GroundingDINO

echo "Installing GroundingDINO package..."
pip install -e .

# ============================================
# Download Model Weights
# ============================================

echo ""
echo "============================================"
echo "Downloading model weights..."
echo "============================================"

mkdir -p weights
cd weights

if [ ! -f "groundingdino_swint_ogc.pth" ]; then
    echo "Downloading GroundingDINO-T weights..."
    wget -q --show-progress https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
else
    echo "Weights already downloaded."
fi

cd ../..

# ============================================
# Verify Installation
# ============================================

echo ""
echo "============================================"
echo "Verifying installation..."
echo "============================================"

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

try:
    from groundingdino.util.inference import load_model
    print('GroundingDINO imported successfully!')
except Exception as e:
    print(f'Error importing GroundingDINO: {e}')
"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To make CUDA_HOME permanent, add this to your ~/.bashrc:"
echo "  echo 'export CUDA_HOME=$CUDA_HOME' >> ~/.bashrc"
echo ""
echo "Run the panel localization script with:"
echo "  python localize_panels.py"
