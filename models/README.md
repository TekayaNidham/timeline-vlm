# models/ — Vision-Language Model Loading

Unified loader for all **37 VLMs** evaluated in the paper. Every model is loaded in zero-shot mode with frozen backbones (no fine-tuning) and returns the same interface: `(model, preprocess, tokenizer)`.

## Files

### `model_loader.py` — Model Registry & Loader

```python
from models import load_model, get_available_models, MODEL_REGISTRY

# Load any model by key
model, preprocess, tokenizer = load_model('clip-vit-b32', device='cuda')

# Encode an image
img_tensor = preprocess(pil_image).unsqueeze(0).to('cuda')
with torch.no_grad():
    embedding = model.encode_image(img_tensor)

# Encode text
tokens = tokenizer(['a photo']).to('cuda')
with torch.no_grad():
    text_emb = model.encode_text(tokens)

# List all 37 models
for name in get_available_models():
    print(name)
```

## Supported Models (37 VLMs)

| Family | Models | Backend |
|---|---|---|
| **CLIP** (9) | RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/16, ViT-B/32, ViT-L/14, ViT-L/14@336 | `openai/CLIP` |
| **EVA-CLIP** (5) | EVA01-g-14, EVA01-g-14-plus, EVA02-B-16, EVA02-L-14, EVA02-L-14-336 | `open_clip` |
| **EVA-CLIP-18B** (3) | EVA-CLIP-8B, EVA-CLIP-8B-plus, EVA-CLIP-18B | `open_clip` |
| **OpenCLIP** (10) | RN50, ViT-B/16 (MetaCLIP, Plus240, QuickGeLU), XLM-RoBERTa-B/32, ViT-B/32, ViT-bigG/14, ViT-G/14, ConvNeXt-XXL | `open_clip` |
| **SigLIP** (3) | ViT-L-16-384, SO400M-14-384, NLLB-large | `open_clip` |
| **CoCa** (1) | CoCa-ViT-L-14 | `open_clip` |
| **MobileCLIP** (1) | MobileCLIP-S1 | `open_clip` |
| **ViTamin** (2) | ViTamin-S, ViTamin-XL-384 | `open_clip` |
| **CLIPA** (1) | CLIPA-ViT-H-14-336 | `open_clip` |
| **ImageBind** (1) | ImageBind-Huge | Custom wrapper |
| **ViT-Lens** (1) | ViT-Lens-2 | Custom wrapper |

## How It Works

Most models (35/37) are loaded via `open_clip.create_model_and_transforms()`. The 9 standard CLIP models use `clip.load()` from the OpenAI CLIP package.

**ImageBind** and **ViT-Lens** have custom APIs, so `model_loader.py` wraps them with a CLIP-compatible interface providing `.encode_image()` and `.encode_text()` methods. These two models require separate installation (see `install_models.sh`).

## Model Keys

Use the model keys from `MODEL_REGISTRY` when referencing models in CLI commands:

```bash
python predict.py --image photo.jpg --model clip-vit-b32
python predict.py --image photo.jpg --model eva-clip-l14-336
python predict.py --image photo.jpg --model openclip-vit-bigg14
```

Run `python run_experiments.py --list_models` to see all available keys.

## Installation

```bash
# Core models (CLIP, OpenCLIP, EVA-CLIP, SigLIP, CoCa, MobileCLIP, ViTamin, CLIPA)
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# Optional: ImageBind and ViT-Lens
bash install_models.sh
```
