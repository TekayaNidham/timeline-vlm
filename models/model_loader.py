"""
Unified model loader for all 37 Vision-Language Models evaluated in the paper.

Supports: CLIP, EVA-CLIP, EVA-CLIP-18B, OpenCLIP, SigLIP, CLIPA, CoCa,
           MobileCLIP, ViTamin, ImageBind, ViT-Lens.

All models are loaded in zero-shot mode with frozen backbones (no fine-tuning).
Each model returns a consistent interface: (model, preprocess, tokenizer).
"""

import os
import sys
import torch


# ──────────────────────────────────────────────────────────────────────
# Complete model registry matching Table 1 of the paper (37 VLMs)
# Format: model_key -> (architecture, backbone/model_name, pretrained, family)
# ──────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    # ── Standard CLIP (9 models) ─────────────────────────────────────
    'clip-rn50':            ('clip', 'RN50',            'openai'),
    'clip-rn101':           ('clip', 'RN101',           'openai'),
    'clip-rn50x4':          ('clip', 'RN50x4',          'openai'),
    'clip-rn50x16':         ('clip', 'RN50x16',         'openai'),
    'clip-rn50x64':         ('clip', 'RN50x64',         'openai'),
    'clip-vit-b16':         ('clip', 'ViT-B/16',        'openai'),
    'clip-vit-b32':         ('clip', 'ViT-B/32',        'openai'),
    'clip-vit-l14':         ('clip', 'ViT-L/14',        'openai'),
    'clip-vit-l14-336':     ('clip', 'ViT-L/14@336px',  'openai'),

    # ── EVA-CLIP (5 models) ──────────────────────────────────────────
    'eva01-clip-g14':       ('open_clip', 'EVA01-g-14',           'laion400m_s11b_b41k'),
    'eva01-clip-g14-plus':  ('open_clip', 'EVA01-g-14-plus',      'merged2b_s11b_b114k'),
    'eva-clip-b16':         ('open_clip', 'EVA02-B-16',            'merged2b_s8b_b131k'),
    'eva-clip-l14':         ('open_clip', 'EVA02-L-14',            'merged2b_s4b_b131k'),
    'eva-clip-l14-336':     ('open_clip', 'EVA02-L-14-336',        'merged2b_s6b_b61k'),

    # ── EVA-CLIP-18B (3 models) ──────────────────────────────────────
    'eva-clip-8b':          ('open_clip', 'EVA02-E-14',            'laion2b_s4b_b115k'),
    'eva-clip-8b-plus':     ('open_clip', 'EVA02-E-14-plus',       'laion2b_s9b_b144k'),
    'eva-clip-18b':         ('open_clip', 'EVA18B-CLIP',            None),  # May need custom loading

    # ── ImageBind (1 model) ──────────────────────────────────────────
    'imagebind':            ('imagebind', 'imagebind_huge',  None),

    # ── CoCa (1 model) ──────────────────────────────────────────────
    'coca-vit-l14':         ('open_clip', 'coca_ViT-L-14', 'mscoco_finetuned_laion2b_s13b_b90k'),

    # ── MobileCLIP (1 model) ─────────────────────────────────────────
    'mobileclip-s1':        ('open_clip', 'MobileCLIP-S1',  'datacompdr'),

    # ── ViTamin (2 models) ───────────────────────────────────────────
    'vitamin-s':            ('open_clip', 'ViTamin-S',      'datacomp1b'),
    'vitamin-xl-384':       ('open_clip', 'ViTamin-XL-384', 'datacomp1b'),

    # ── OpenCLIP variants (10 models) ────────────────────────────────
    'openclip-rn50-quickgelu':      ('open_clip', 'RN50-quickgelu',                 'yfcc15m'),
    'openclip-vit-b16-metaclip':    ('open_clip', 'ViT-B-16',                       'metaclip_400m'),
    'openclip-vit-b16-plus-240':    ('open_clip', 'ViT-B-16-plus-240',              'laion400m_e32'),
    'openclip-vit-b16-quickgelu':   ('open_clip', 'ViT-B-16-quickgelu',             'dfn2b'),
    'openclip-xlm-roberta-b32':     ('open_clip', 'xlm-roberta-base-ViT-B-32',      'laion5b_s13b_b90k'),
    'openclip-vit-b32':             ('open_clip', 'ViT-B-32',                        'commonpool_xl_clip_s13b_b90k'),
    'openclip-vit-bigg14':          ('open_clip', 'ViT-bigG-14',                     'laion2b_s39b_b160k'),
    'openclip-vit-bigg14-quickgelu': ('open_clip', 'ViT-bigG-14-quickgelu',          'metaclip_fullcc'),
    'openclip-vit-g14':             ('open_clip', 'ViT-g-14',                        'laion2b_s34b_b88k'),
    'openclip-convnext-xxlarge':    ('open_clip', 'convnext_xxlarge',                'laion2b_s34b_b82k_augreg'),

    # ── CLIPA (1 model) ─────────────────────────────────────────────
    'clipa-vit-h14-336':    ('open_clip', 'ViT-H-14-CLIPA-336', 'datacomp1b'),

    # ── SigLIP (3 models) ───────────────────────────────────────────
    'siglip-vit-l16-384':   ('open_clip', 'ViT-L-16-SigLIP-384',        'webli'),
    'siglip-so400m-384':    ('open_clip', 'ViT-SO400M-14-SigLIP-384',   'webli'),
    'siglip-nllb-large':    ('open_clip', 'nllb-clip-large-siglip',      'v1'),

    # ── ViT-Lens (1 model) ──────────────────────────────────────────
    'vit-lens':             ('vit_lens', 'ViT-Lens-L',  None),
}


def load_model(model_name, device='cuda'):
    """
    Load a vision-language model by name.

    Args:
        model_name: Key from MODEL_REGISTRY (case-insensitive)
        device: 'cuda' or 'cpu'

    Returns:
        (model, preprocess, tokenizer) tuple with consistent interface:
        - model.encode_image(tensor) -> normalized embeddings
        - model.encode_text(tokens) -> normalized embeddings
        - tokenizer(list_of_strings) -> token tensor
        - preprocess(PIL.Image) -> tensor
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}\n"
            f"Available: {', '.join(sorted(MODEL_REGISTRY.keys()))}"
        )

    family, backbone, pretrained = MODEL_REGISTRY[model_name]

    if family == 'clip':
        return _load_clip(backbone, device)
    elif family == 'open_clip':
        return _load_open_clip(backbone, pretrained, device)
    elif family == 'imagebind':
        return _load_imagebind(device)
    elif family == 'vit_lens':
        return _load_vit_lens(device)
    else:
        raise ValueError(f"Unknown model family: {family}")


def _load_clip(backbone, device):
    """Load standard OpenAI CLIP model."""
    import clip
    model, preprocess = clip.load(backbone, device=device)
    model.eval()
    tokenizer = clip.tokenize
    return model, preprocess, tokenizer


def _load_open_clip(backbone, pretrained, device):
    """Load any OpenCLIP-compatible model (EVA-CLIP, SigLIP, CoCa, etc.)."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        backbone, pretrained=pretrained
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(backbone)
    return model, preprocess, tokenizer


def _load_imagebind(device):
    """Load ImageBind model with CLIP-compatible wrapper."""
    imagebind_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'models', 'ImageBind'
    )
    if os.path.exists(imagebind_path):
        sys.path.insert(0, imagebind_path)

    try:
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        from imagebind import data as ib_data

        model = imagebind_model.imagebind_huge(pretrained=True)
        model = model.to(device).eval()

        class ImageBindWrapper:
            """Wrapper to provide CLIP-compatible encode_image/encode_text."""
            def __init__(self, model, device):
                self.model = model
                self.device = device

            def encode_image(self, images):
                inputs = {ModalityType.VISION: images}
                with torch.no_grad():
                    embeddings = self.model(inputs)
                emb = embeddings[ModalityType.VISION]
                return emb / emb.norm(dim=-1, keepdim=True)

            def encode_text(self, texts):
                inputs = {ModalityType.TEXT: texts}
                with torch.no_grad():
                    embeddings = self.model(inputs)
                emb = embeddings[ModalityType.TEXT]
                return emb / emb.norm(dim=-1, keepdim=True)

        wrapped = ImageBindWrapper(model, device)

        def preprocess(image):
            return ib_data.load_and_transform_vision_data([image], device)

        def tokenizer(texts):
            if isinstance(texts, str):
                texts = [texts]
            return ib_data.load_and_transform_text(texts, device)

        return wrapped, preprocess, tokenizer

    except ImportError:
        raise ImportError(
            "ImageBind not found. Install with: bash install_models.sh"
        )


def _load_vit_lens(device):
    """Load ViT-Lens model with CLIP-compatible wrapper."""
    vitlens_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'models', 'ViT-Lens'
    )
    if os.path.exists(vitlens_path):
        sys.path.insert(0, vitlens_path)

    try:
        from mm_vit_lens import ViTLens
        from open_clip import ModalityType

        model = ViTLens(modality_loaded=[ModalityType.IMAGE, ModalityType.TEXT])
        model = model.to(device).eval()

        class ViTLensWrapper:
            def __init__(self, model):
                self.model = model

            def encode_image(self, images):
                inputs = {ModalityType.IMAGE: images}
                with torch.no_grad():
                    output = self.model.encode(inputs, normalize=True)
                return output[ModalityType.IMAGE]

            def encode_text(self, texts):
                inputs = {ModalityType.TEXT: texts}
                with torch.no_grad():
                    output = self.model.encode(inputs, normalize=True)
                return output[ModalityType.TEXT]

        wrapped = ViTLensWrapper(model)

        def preprocess(image):
            return image

        def tokenizer(texts):
            return texts

        return wrapped, preprocess, tokenizer

    except ImportError:
        raise ImportError(
            "ViT-Lens not found. Install with: bash install_models.sh"
        )


def get_available_models():
    """Return sorted list of all model keys."""
    return sorted(MODEL_REGISTRY.keys())


def get_model_info(model_name=None):
    """Get info about one or all models."""
    if model_name:
        model_name = model_name.lower()
        if model_name in MODEL_REGISTRY:
            family, backbone, pretrained = MODEL_REGISTRY[model_name]
            return {
                'key': model_name,
                'family': family,
                'backbone': backbone,
                'pretrained': pretrained,
            }
        return None

    return {
        k: {'family': v[0], 'backbone': v[1], 'pretrained': v[2]}
        for k, v in sorted(MODEL_REGISTRY.items())
    }


def get_model_family(model_name):
    """Get the family of a model (clip, open_clip, imagebind, vit_lens)."""
    model_name = model_name.lower()
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name][0]
    return None


def is_clip_family(model_name):
    """Check if model uses CLIP tokenizer (vs OpenCLIP)."""
    model_name = model_name.lower()
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name][0] == 'clip'
    return False


if __name__ == '__main__':
    print(f"Available models ({len(MODEL_REGISTRY)} total):\n")
    for key in get_available_models():
        family, backbone, pretrained = MODEL_REGISTRY[key]
        pt = pretrained or 'default'
        print(f"  {key:<40} {family:<10} {backbone:<35} {pt}")
