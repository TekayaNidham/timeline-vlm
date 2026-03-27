#!/bin/bash
set -e

echo "=================================================="
echo "Installing dependencies for Timeline-VLM"
echo "=================================================="

# Core dependencies
echo "[1/5] Installing core dependencies..."
pip install -r requirements.txt

# OpenAI CLIP
echo "[2/5] Installing OpenAI CLIP..."
pip install git+https://github.com/openai/CLIP.git

# EVA-CLIP (original BAAI repo — used for paper results)
echo "[3/5] Installing EVA-CLIP (BAAI repo)..."
if [ ! -d "models/EVA" ]; then
    echo "  Cloning EVA repository..."
    git clone https://github.com/baaivision/EVA.git models/EVA
    cd models/EVA/EVA-CLIP
    pip install -e . 2>/dev/null || echo "  EVA-CLIP install failed (non-critical, will fall back to open_clip)"
    cd ../../..
else
    echo "  EVA already present"
fi

# ImageBind (optional — needed only for ImageBind model)
echo "[4/5] Installing ImageBind (optional)..."
if [ ! -d "models/ImageBind" ]; then
    echo "  Cloning ImageBind..."
    git clone https://github.com/facebookresearch/ImageBind.git models/ImageBind
    cd models/ImageBind
    pip install -e . 2>/dev/null || echo "  ImageBind install failed (non-critical)"
    cd ../..
else
    echo "  ImageBind already present"
fi

# ViT-Lens (optional — needed only for ViT-Lens model)
echo "[5/5] Installing ViT-Lens (optional)..."
if [ ! -d "models/ViT-Lens" ]; then
    echo "  Cloning ViT-Lens..."
    git clone https://github.com/TencentARC/ViT-Lens.git models/ViT-Lens
    cd models/ViT-Lens
    pip install -e . 2>/dev/null || echo "  ViT-Lens install failed (non-critical)"
    cd ../..
else
    echo "  ViT-Lens already present"
fi

echo ""
echo "=================================================="
echo "Installation complete!"
echo "=================================================="
echo ""
echo "Core models (CLIP, OpenCLIP, SigLIP, CoCa,"
echo "MobileCLIP, ViTamin, CLIPA) are available via open_clip."
echo ""
echo "EVA-CLIP models use the original BAAI repository."
echo "If eva_clip install failed, they will fall back to open_clip."
echo ""
echo "Optional models (ImageBind, ViT-Lens) require their"
echo "respective repositories in models/."
echo ""
echo "Quick test:"
echo "  python run_experiments.py --config configs/lightweight_test.yaml --device cpu"
echo ""
