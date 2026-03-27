"""
Embedding generation pipeline.

Generates image and text (time) embeddings for all models,
with support for caching to disk.
"""

import os
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def generate_time_embeddings(model, tokenizer, model_name, prompt_template,
                             years=None, device='cuda', batch_size=64):
    """
    Generate text embeddings for year prompts.

    Args:
        model: VLM with encode_text method
        tokenizer: Text tokenizer
        model_name: Model key (for CLIP vs OpenCLIP dispatch)
        prompt_template: Template string with {year} placeholder
        years: List of years (default: 1700-2024)
        device: Compute device
        batch_size: Batch size for encoding

    Returns:
        embeddings: np.ndarray of shape (num_years, embed_dim)
        years: List of years
    """
    from models.model_loader import is_clip_family

    if years is None:
        years = list(range(1700, 2025))

    all_embeddings = []

    for i in range(0, len(years), batch_size):
        batch_years = years[i:i + batch_size]
        prompts = [prompt_template.format(year=y) for y in batch_years]

        if is_clip_family(model_name):
            import clip
            tokens = clip.tokenize(prompts).to(device)
        else:
            tokens = tokenizer(prompts).to(device)

        with torch.no_grad():
            text_emb = model.encode_text(tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        all_embeddings.append(text_emb.cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    return embeddings, years


def generate_image_embeddings(model, preprocess, image_paths, device='cuda',
                              batch_size=32):
    """
    Generate image embeddings for a list of image paths.

    Args:
        model: VLM with encode_image method
        preprocess: Image preprocessing transform
        image_paths: List of image file paths
        device: Compute device
        batch_size: Batch size for encoding

    Returns:
        embeddings: np.ndarray of shape (num_images, embed_dim)
        valid_indices: List of indices that were successfully processed
        timing: Dict with per-image timing stats in ms
    """
    all_embeddings = []
    valid_indices = []
    times_ms = []

    for i in tqdm(range(0, len(image_paths), batch_size),
                  desc="Encoding images"):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []
        batch_indices = []

        for j, path in enumerate(batch_paths):
            try:
                image = Image.open(path).convert('RGB')
                tensor = preprocess(image).unsqueeze(0)
                batch_tensors.append(tensor)
                batch_indices.append(i + j)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue

        if not batch_tensors:
            continue

        images = torch.cat(batch_tensors, dim=0).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            image_emb = model.encode_image(images)
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        all_embeddings.append(image_emb.cpu().numpy())
        valid_indices.extend(batch_indices)
        times_ms.append(elapsed_ms / len(batch_tensors))

    if not all_embeddings:
        return np.array([]), [], {}

    embeddings = np.vstack(all_embeddings)

    timing = {
        'avg_ms': float(np.mean(times_ms)) if times_ms else 0,
        'total_images': len(valid_indices),
    }

    return embeddings, valid_indices, timing


def save_embeddings(embeddings, years_or_labels, output_dir, prefix=''):
    """Save embeddings and labels to disk."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{prefix}_" if prefix else ""

    np.save(os.path.join(output_dir, f'{prefix}embeddings.npy'), embeddings)

    labels_path = os.path.join(output_dir, f'{prefix}labels.txt')
    with open(labels_path, 'w') as f:
        for label in years_or_labels:
            f.write(f"{label}\n")


def load_embeddings(embeddings_path, labels_path):
    """Load embeddings and labels from disk."""
    embeddings = np.load(embeddings_path)
    with open(labels_path, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    return embeddings, labels


def load_precomputed_embeddings(embeddings_dir, model_name):
    """
    Load precomputed embeddings for a model.

    Supports the existing directory structure:
    - CLIP: encodings/{timeline,image}_embeddings.npy
    - EVA-CLIP: encodings/eva/eva_{timeline,image}_embeddings.npy

    Returns:
        dict with keys: timeline_emb, image_emb, timeline_years, image_years
    """
    embeddings_dir = Path(embeddings_dir)

    if 'eva' in model_name.lower():
        subdir = embeddings_dir / 'eva'
        timeline_emb = np.load(subdir / 'eva_timeline_embeddings.npy')
        image_emb = np.load(subdir / 'eva_image_embeddings.npy')
        with open(subdir / 'eva_labels_timeline.txt') as f:
            timeline_years = [int(l.strip()) for l in f]
        with open(subdir / 'labels.txt') as f:
            image_years = [int(l.strip()) for l in f]
    else:
        timeline_emb = np.load(embeddings_dir / 'timeline_embeddings.npy')
        image_emb = np.load(embeddings_dir / 'image_embeddings.npy')
        with open(embeddings_dir / 'timeline_labels.txt') as f:
            timeline_years = [int(l.strip()) for l in f]
        with open(embeddings_dir / 'labels.txt') as f:
            image_years = [int(l.strip()) for l in f]

    return {
        'timeline_emb': timeline_emb,
        'image_emb': image_emb,
        'timeline_years': timeline_years,
        'image_years': np.array(image_years),
    }
