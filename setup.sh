#!/bin/bash
# Traced — SAM 2 Setup
# Run from the traced repo root

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT_DIR="$SCRIPT_DIR/checkpoints"

echo "=== Traced: SAM 2 Setup ==="

# Create conda env if it doesn't exist
if ! conda info --envs 2>/dev/null | grep -q "traced"; then
    echo "Creating conda environment 'traced'..."
    conda create -n traced python=3.12 -y
fi

echo "Activating traced environment..."
eval "$(conda shell.bash hook)"
conda activate traced

# Install PyTorch with CUDA
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install SAM 2
echo "Installing SAM 2..."
pip install sam-2 || {
    echo "pip install failed, trying from source..."
    git clone https://github.com/facebookresearch/sam2.git "$SCRIPT_DIR/.sam2-src"
    cd "$SCRIPT_DIR/.sam2-src"
    pip install -e ".[notebooks]"
    cd "$SCRIPT_DIR"
}

# Download checkpoints into repo
echo "Downloading SAM 2 checkpoints..."
mkdir -p "$CKPT_DIR"
cd "$CKPT_DIR"

if [ ! -f sam2.1_hiera_large.pt ]; then
    echo "Downloading sam2.1_hiera_large.pt (~900MB)..."
    wget -q --show-progress "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
fi

if [ ! -f sam2.1_hiera_base_plus.pt ]; then
    echo "Downloading sam2.1_hiera_base_plus.pt (~320MB)..."
    wget -q --show-progress "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
fi

# Other dependencies
echo "Installing pipeline dependencies..."
pip install opencv-python-headless pillow scipy

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run extraction:"
echo "  conda activate traced"
echo "  cd $SCRIPT_DIR"
echo "  python extract-sam2.py --image examples/szm-reference.jpg --output szm-extraction.json"
