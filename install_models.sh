#!/bin/bash

echo "Installing model-specific dependencies..."

# Create models directory if it doesn't exist
mkdir -p models

# Install EVA-CLIP
echo "Installing EVA-CLIP..."
cd models
if [ ! -d "EVA-CLIP" ]; then
    git clone https://github.com/baaivision/EVA.git EVA-CLIP
    cd EVA-CLIP
    pip install -e .
    cd ..
else
    echo "EVA-CLIP already installed"
fi
cd ..

# Install ImageBind
echo "Installing ImageBind..."
cd models
if [ ! -d "ImageBind" ]; then
    git clone https://github.com/facebookresearch/ImageBind.git
    cd ImageBind
    pip install -e .
    cd ..
else
    echo "ImageBind already installed"
fi
cd ..

# Install ViT-Lens
echo "Installing ViT-Lens..."
cd models
if [ ! -d "ViT-Lens" ]; then
    git clone https://github.com/TencentARC/ViT-Lens.git
    cd ViT-Lens
    pip install -e .
    cd ..
else
    echo "ViT-Lens already installed"
fi
cd ..

# Download model weights if needed
echo "Setting up model weights..."
mkdir -p weights

# EVA-CLIP weights
if [ ! -f "weights/EVA02_CLIP_L_psz14_s4B.pt" ]; then
    echo "Downloading EVA-CLIP weights..."
    # Add wget commands for weights here
fi

echo "Installation complete!"
echo "Note: Some models may require additional manual setup. Please refer to their respective repositories."