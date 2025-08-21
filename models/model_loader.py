"""
Model loader for various Vision-Language Models
Handles loading and initialization of CLIP, EVA-CLIP, OpenCLIP, ImageBind, and ViT-Lens
"""

import os
import sys
import torch


def load_model(model_name, device='cuda'):
    """
    Load a vision-language model by name
    
    Args:
        model_name: Name of the model to load
        device: Device to load model on
        
    Returns:
        model, preprocess, tokenizer
    """
    model_name = model_name.lower()
    
    # Standard CLIP models
    if model_name in ['clip-vit-b32', 'clip-vit-b16', 'clip-vit-l14', 'clip-vit-l14-336',
                      'clip-rn50', 'clip-rn101', 'clip-rn50x4', 'clip-rn50x16', 'clip-rn50x64']:
        import clip
        model_type = model_name.replace('clip-', '').replace('-', '/').upper()
        if '336' in model_type:
            model_type = model_type.replace('/336', '@336px')
        model, preprocess = clip.load(model_type, device=device)
        tokenizer = clip.tokenize
        return model, preprocess, tokenizer
    
    # EVA-CLIP models
    elif 'eva' in model_name:
        # Add EVA-CLIP path to sys.path
        eva_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'EVA-CLIP')
        if os.path.exists(eva_path):
            sys.path.insert(0, eva_path)
        
        try:
            from eva_clip import create_model_and_transforms, get_tokenizer
            
            # Map model names to EVA-CLIP names
            eva_model_map = {
                'eva-clip-b16': 'EVA02-CLIP-B-16',
                'eva-clip-l14': 'EVA02-CLIP-L-14',
                'eva-clip-l14-336': 'EVA02-CLIP-L-14-336',
                'eva01-clip-g14': 'EVA01-CLIP-g-14',
                'eva01-clip-g14-plus': 'EVA01-CLIP-g-14-plus',
                'eva-clip-8b': 'EVA-CLIP-8B',
                'eva-clip-8b-plus': 'EVA-CLIP-8B-plus',
                'eva-clip-18b': 'EVA-CLIP-18B'
            }
            
            backbone = eva_model_map.get(model_name, 'EVA02-CLIP-L-14')
            
            # Set pretrained path based on model
            if '18b' in model_name.lower():
                pretrained = "EVA_CLIP_18B_psz14_s6B.fp16.pt"
            else:
                pretrained = "eva_clip"
            
            model, _, preprocess = create_model_and_transforms(
                backbone, 
                pretrained, 
                force_custom_clip=True
            )
            model = model.to(device)
            tokenizer = get_tokenizer(backbone)
            
            return model, preprocess, tokenizer
            
        except ImportError:
            raise ImportError("EVA-CLIP not found. Please run install_models.sh")
    
    # OpenCLIP models
    elif 'openclip' in model_name or any(x in model_name for x in ['vitamin', 'coca', 'convnext']):
        import open_clip
        
        # Map model names to OpenCLIP format
        openclip_models = {
            'openclip-vit-b32': ('ViT-B-32', 'laion2b_s34b_b79k'),
            'openclip-vit-b16': ('ViT-B-16', 'laion2b_s34b_b88k'),
            'openclip-vit-l14': ('ViT-L-14', 'laion2b_s32b_b82k'),
            'openclip-convnext-xxlarge': ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg'),
            'vitamin-s': ('ViTamin-S', 'datacomp1b'),
            'vitamin-xl-384': ('ViTamin-XL-384', 'datacomp1b'),
            'coca-vit-l14': ('coca_ViT-L-14', 'mscoco_finetuned_laion2b_s13b_b90k'),
        }
        
        if model_name in openclip_models:
            arch, pretrained = openclip_models[model_name]
        else:
            # Default
            arch, pretrained = 'ViT-B-32', 'laion2b_s34b_b79k'
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, 
            pretrained=pretrained
        )
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer(arch)
        
        return model, preprocess, tokenizer
    
    # ImageBind
    elif 'imagebind' in model_name:
        imagebind_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'ImageBind')
        if os.path.exists(imagebind_path):
            sys.path.insert(0, imagebind_path)
        
        try:
            from imagebind.models import imagebind_model
            from imagebind import data
            
            model = imagebind_model.imagebind_huge(pretrained=True)
            model = model.to(device).eval()
            
            # Create wrapper for consistent interface
            class ImageBindWrapper:
                def __init__(self, model):
                    self.model = model
                    
                def encode_image(self, images):
                    # Assuming images is a tensor
                    inputs = {
                        imagebind_model.ModalityType.VISION: images
                    }
                    with torch.no_grad():
                        embeddings = self.model(inputs)
                    return embeddings[imagebind_model.ModalityType.VISION]
                
                def encode_text(self, texts):
                    # texts should be tokenized
                    inputs = {
                        imagebind_model.ModalityType.TEXT: texts
                    }
                    with torch.no_grad():
                        embeddings = self.model(inputs)
                    return embeddings[imagebind_model.ModalityType.TEXT]
            
            wrapped_model = ImageBindWrapper(model)
            
            # Preprocess function
            def preprocess(image):
                # ImageBind expects paths, so we need a custom preprocessor
                return data.load_and_transform_vision_data([image], device)
            
            # Tokenizer function
            def tokenizer(texts):
                return data.load_and_transform_text(texts, device)
            
            return wrapped_model, preprocess, tokenizer
            
        except ImportError:
            raise ImportError("ImageBind not found. Please run install_models.sh")
    
    # ViT-Lens
    elif 'vit-lens' in model_name:
        vitlens_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'ViT-Lens')
        if os.path.exists(vitlens_path):
            sys.path.insert(0, vitlens_path)
        
        try:
            from mm_vit_lens import ViTLens
            from open_clip import ModalityType
            
            model = ViTLens(modality_loaded=[ModalityType.IMAGE, ModalityType.TEXT])
            model = model.to(device)
            
            # Create wrapper for consistent interface
            class ViTLensWrapper:
                def __init__(self, model):
                    self.model = model
                    
                def encode_image(self, images):
                    # ViT-Lens expects paths
                    inputs = {ModalityType.IMAGE: images}
                    with torch.no_grad():
                        output = self.model.encode(inputs, normalize=True)
                    return output[ModalityType.IMAGE]
                
                def encode_text(self, texts):
                    inputs = {ModalityType.TEXT: texts}
                    with torch.no_grad():
                        output = self.model.encode(inputs, normalize=True)
                    return output[ModalityType.TEXT]
            
            wrapped_model = ViTLensWrapper(model)
            
            # Minimal preprocess and tokenizer
            def preprocess(image):
                return image  # ViT-Lens handles preprocessing
            
            def tokenizer(texts):
                return texts  # ViT-Lens handles tokenization
            
            return wrapped_model, preprocess, tokenizer
            
        except ImportError:
            raise ImportError("ViT-Lens not found. Please run install_models.sh")
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_available_models():
    """Return list of available model names"""
    models = [
        # CLIP
        'clip-vit-b32', 'clip-vit-b16', 'clip-vit-l14', 'clip-vit-l14-336',
        'clip-rn50', 'clip-rn101', 'clip-rn50x4', 'clip-rn50x16', 'clip-rn50x64',
        
        # EVA-CLIP
        'eva-clip-b16', 'eva-clip-l14', 'eva-clip-l14-336',
        'eva01-clip-g14', 'eva01-clip-g14-plus',
        'eva-clip-8b', 'eva-clip-8b-plus', 'eva-clip-18b',
        
        # OpenCLIP
        'openclip-vit-b32', 'openclip-vit-b16', 'openclip-vit-l14',
        'openclip-convnext-xxlarge', 'vitamin-s', 'vitamin-xl-384',
        'coca-vit-l14',
        
        # Others
        'imagebind', 'vit-lens'
    ]
    return models