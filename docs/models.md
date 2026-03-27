# Supported Models (37 VLMs)

All models are loaded in zero-shot mode with frozen backbones. Each returns a consistent interface: `(model, preprocess, tokenizer)`.

## Model Families

### CLIP (9 models) — OpenAI

Standard CLIP models loaded via `clip.load()`.

| Key | Architecture | Input |
|---|---|---|
| `clip-rn50` | ResNet-50 | 224px |
| `clip-rn101` | ResNet-101 | 224px |
| `clip-rn50x4` | ResNet-50x4 | 288px |
| `clip-rn50x16` | ResNet-50x16 | 384px |
| `clip-rn50x64` | ResNet-50x64 | 448px |
| `clip-vit-b16` | ViT-B/16 | 224px |
| `clip-vit-b32` | ViT-B/32 | 224px |
| `clip-vit-l14` | ViT-L/14 | 224px |
| `clip-vit-l14-336` | ViT-L/14 | 336px |

### EVA-CLIP (8 models) — BAAI

Loaded from the **original BAAI EVA repository** (`eva_clip` package). Falls back to `open_clip` with a warning if `eva_clip` is not installed.

| Key | Architecture | Pretrained |
|---|---|---|
| `eva01-clip-g14` | EVA01-g-14 | LAION-400M |
| `eva01-clip-g14-plus` | EVA01-g-14-plus | Merged-2B |
| `eva-clip-b16` | EVA02-B-16 | Merged-2B |
| `eva-clip-l14` | EVA02-L-14 | Merged-2B |
| `eva-clip-l14-336` | EVA02-L-14-336 | Merged-2B |
| `eva-clip-8b` | EVA02-E-14 | LAION-2B |
| `eva-clip-8b-plus` | EVA02-E-14-plus | LAION-2B |
| `eva-clip-18b` | EVA-CLIP-18B | — |

### OpenCLIP (10 models)

| Key | Architecture | Pretrained |
|---|---|---|
| `openclip-rn50-quickgelu` | RN50-quickgelu | YFCC-15M |
| `openclip-vit-b16-metaclip` | ViT-B-16 | MetaCLIP-400M |
| `openclip-vit-b16-plus-240` | ViT-B-16-plus-240 | LAION-400M |
| `openclip-vit-b16-quickgelu` | ViT-B-16-quickgelu | DFN-2B |
| `openclip-xlm-roberta-b32` | XLM-RoBERTa-B-32 | LAION-5B |
| `openclip-vit-b32` | ViT-B-32 | CommonPool-XL |
| `openclip-vit-bigg14` | ViT-bigG-14 | LAION-2B |
| `openclip-vit-bigg14-quickgelu` | ViT-bigG-14-quickgelu | MetaCLIP-FullCC |
| `openclip-vit-g14` | ViT-g-14 | LAION-2B |
| `openclip-convnext-xxlarge` | ConvNeXt-XXL | LAION-2B |

### SigLIP (3 models)

| Key | Architecture |
|---|---|
| `siglip-vit-l16-384` | ViT-L-16-SigLIP-384 |
| `siglip-so400m-384` | ViT-SO400M-14-SigLIP-384 |
| `siglip-nllb-large` | NLLB-CLIP-large-SigLIP |

### Others (7 models)

| Key | Family | Architecture |
|---|---|---|
| `coca-vit-l14` | CoCa | CoCa-ViT-L-14 |
| `mobileclip-s1` | MobileCLIP | MobileCLIP-S1 |
| `vitamin-s` | ViTamin | ViTamin-S |
| `vitamin-xl-384` | ViTamin | ViTamin-XL-384 |
| `clipa-vit-h14-336` | CLIPA | ViT-H-14-CLIPA-336 |
| `imagebind` | ImageBind | ImageBind-Huge |
| `vit-lens` | ViT-Lens | ViT-Lens-L |

## Usage

```python
from models import load_model, get_available_models

# Load any model
model, preprocess, tokenizer = load_model('clip-vit-b32', device='cuda')

# List all keys
for name in get_available_models():
    print(name)
```

## Installation

```bash
# Core (CLIP + OpenCLIP-based models)
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# EVA-CLIP (original BAAI repo)
git clone https://github.com/baaivision/EVA.git models/EVA
pip install -e models/EVA/EVA-CLIP

# ImageBind + ViT-Lens
bash install_models.sh
```
