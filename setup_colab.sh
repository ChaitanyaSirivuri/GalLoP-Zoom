# 1. Install GroundingDINO
echo "Cloning GroundingDINO..."
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/

echo "Patching GroundingDINO for PyTorch 2.x..."
# Fix for "const at::DeprecatedTypeProperties" error in ms_deform_attn_cuda.cu
# Replaces value.type() with value.scalar_type()
sed -i 's/value.type()/value.scalar_type()/g' groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu

echo "Installing dependencies..."
# Ensure CUDA_HOME is set
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
echo "CUDA_HOME is set to $CUDA_HOME"
echo "Checking nvcc..."
nvcc --version

pip install -e .

mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..