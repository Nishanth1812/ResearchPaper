#!/bin/bash
# DigitalOcean GPU Droplet Setup Script for DBGDGM Training
# Run this script on a fresh DigitalOcean GPU Droplet (Ubuntu 22.04)
# Uses uv as package manager (with pip fallback)

set -e  # Exit on error

# Configuration - adjust these paths as needed
STORAGE_VOLUME="/mnt/storage"  # Your attached storage volume mount point
REPO_DIR="$HOME/ResearchPaper"
TRAINING_DIR="$REPO_DIR/training_digitalocean"

echo "=============================================="
echo "🚀 DBGDGM Training Environment Setup"
echo "   DigitalOcean GPU Droplet (RTX 6000 Ada)"
echo "=============================================="

# Update system
echo ""
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install essentials
echo ""
echo "� Installing essentials..."
sudo apt-get install -y python3.10 python3.10-venv git htop nvtop unzip curl

# Install uv (fast Python package manager)
echo ""
echo "⚡ Installing uv package manager..."
if command -v uv &> /dev/null; then
    echo "   uv already installed"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Verify uv installation
if command -v uv &> /dev/null; then
    echo "   ✅ uv installed: $(uv --version)"
    USE_UV=true
else
    echo "   ⚠️  uv installation failed, falling back to pip"
    USE_UV=false
fi

# Create virtual environment
echo ""
echo "📁 Creating virtual environment..."
cd "$REPO_DIR"

if [ "$USE_UV" = true ]; then
    uv venv .venv --python 3.10
    source .venv/bin/activate
else
    python3.10 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
fi

# Install dependencies
echo ""
echo "� Installing dependencies..."

if [ "$USE_UV" = true ]; then
    # Install PyTorch with CUDA 12.1
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install other dependencies
    uv pip install \
        numpy==1.23.4 \
        scipy==1.9.3 \
        scikit-learn==1.1.3 \
        networkx==2.8.8 \
        nibabel==4.0.2 \
        nilearn==0.9.2 \
        pandas==1.5.1 \
        matplotlib==3.6.2 \
        tqdm==4.64.1 \
        lxml==4.9.1
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install \
        numpy==1.23.4 \
        scipy==1.9.3 \
        scikit-learn==1.1.3 \
        networkx==2.8.8 \
        nibabel==4.0.2 \
        nilearn==0.9.2 \
        pandas==1.5.1 \
        matplotlib==3.6.2 \
        tqdm==4.64.1 \
        lxml==4.9.1
fi

# Extract dataset if exists
echo ""
echo "📊 Checking for dataset..."
if [ -f "$REPO_DIR/dataset.zip" ]; then
    echo "   Found dataset.zip, extracting..."
    # Note: dataset.zip already contains data/ folder structure
    unzip -o "$REPO_DIR/dataset.zip" -d "$REPO_DIR/"
    echo "   ✅ Dataset extracted to $REPO_DIR/data/"
else
    echo "   ⚠️  dataset.zip not found in $REPO_DIR"
fi

# Setup storage volume for models
echo ""
echo "💾 Setting up storage volume..."
if [ -d "$STORAGE_VOLUME" ]; then
    mkdir -p "$STORAGE_VOLUME/dbgdgm_models"
    echo "   ✅ Models will be saved to: $STORAGE_VOLUME/dbgdgm_models"
else
    echo "   ⚠️  Storage volume not found at $STORAGE_VOLUME"
    echo "   Models will be saved locally in the repo"
fi

# Verify GPU detection
echo ""
echo "🎮 Verifying GPU detection..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=============================================="
echo "✅ Setup complete!"
echo ""
echo "Activate environment:"
echo "  source $REPO_DIR/.venv/bin/activate"
echo ""
echo "Run training (models saved to storage volume):"
echo "  cd $TRAINING_DIR"
echo "  python do_train.py --dataset hcp --categorical-dim 8 --trial 1 \\"
echo "      --data-dir $REPO_DIR/data --output-dir $STORAGE_VOLUME/dbgdgm_models --fast"
echo ""
echo "Or run in background:"
echo "  cd $TRAINING_DIR"
echo "  nohup python do_train.py --dataset hcp --categorical-dim 8 --trial 1 \\"
echo "      --data-dir $REPO_DIR/data --output-dir $STORAGE_VOLUME/dbgdgm_models --fast > training.log 2>&1 &"
echo "=============================================="
