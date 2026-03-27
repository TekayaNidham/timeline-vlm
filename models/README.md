# models/

Unified loader for all **37 VLMs** evaluated in the paper. Every model returns the same interface: `(model, preprocess, tokenizer)`.

```python
from models import load_model, get_available_models, MODEL_REGISTRY

model, preprocess, tokenizer = load_model('clip-vit-b32', device='cuda')
```

**Loading backends:**
- **CLIP** (9): `clip.load()` from OpenAI CLIP
- **EVA-CLIP** (8): `eva_clip` from the original BAAI EVA repo (falls back to `open_clip`)
- **OpenCLIP, SigLIP, CoCa, MobileCLIP, ViTamin, CLIPA** (17): `open_clip`
- **ImageBind, ViT-Lens** (2): Custom wrappers with CLIP-compatible API

See [`docs/models.md`](../docs/models.md) for the full model list and installation instructions.
