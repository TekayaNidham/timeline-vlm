"""
Predict the year of first appearance for images using temporal VLM analysis.

Supports all models, methods, and configurations from the paper.
Designed for use in pipelines and interactive exploration.

Usage:
    # Single image with default settings (CLIP ViT-B/32, Bézier R^S)
    python predict.py --image photo.jpg

    # Batch prediction on a directory
    python predict.py --image_dir my_photos/ --model eva-clip-l14-336

    # Time probing method with specific prompt
    python predict.py --image photo.jpg --method time_probing --prompt P7

    # Bézier with custom settings
    python predict.py --image photo.jpg --method bezier --reduce_dim 13 --bezier_method interpolation

    # UMAP method
    python predict.py --image photo.jpg --method umap --model clip-vit-b32

    # Use precomputed embeddings (fast, no dataset needed)
    python predict.py --image photo.jpg --embeddings_path encodings

    # JSON output for pipeline integration
    python predict.py --image photo.jpg --output json
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model_loader import load_model, is_clip_family, get_available_models
from utils.prompts import get_prompt_templates
from utils.metrics import calculate_TAI


def predict_time_probing(model, preprocess, tokenizer, model_name, image,
                         prompt_template, years, device):
    """Predict year via time probing (dot-product similarity)."""
    # Encode time embeddings
    all_text_emb = []
    for i in range(0, len(years), 64):
        batch = years[i:i+64]
        prompts = [prompt_template.format(year=y) for y in batch]
        if is_clip_family(model_name):
            import clip
            tokens = clip.tokenize(prompts).to(device)
        else:
            tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_text_emb.append(emb.cpu().numpy())
    time_emb = np.vstack(all_text_emb)

    # Encode image
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = model.encode_image(img_tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        img_emb = img_emb.cpu().numpy()

    # Similarity
    sims = (100.0 * (img_emb @ time_emb.T)).flatten()
    top_indices = np.argsort(sims)[::-1]

    return {
        'predicted_year': int(years[top_indices[0]]),
        'confidence_scores': {
            int(years[top_indices[i]]): float(sims[top_indices[i]])
            for i in range(min(5, len(top_indices)))
        },
        'all_similarities': sims,
    }


def predict_with_embeddings(model, preprocess, image, precomputed,
                            method, device, bezier_method='interpolation',
                            reduce_dim=13, num_control_points=200):
    """Predict year using precomputed timeline embeddings + a timeline method."""
    from evaluation.timeline_umap import UMAPTimeline
    from evaluation.timeline_bezier import BezierTimeline

    timeline_emb = precomputed['timeline_emb']
    timeline_years = precomputed['timeline_years']

    # Encode query image
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = model.encode_image(img_tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        img_emb = img_emb.cpu().numpy()

    if method == 'umap':
        timeline = UMAPTimeline()
        timeline.fit(timeline_emb, timeline_years, model_name='clip-vit-b32')
        predictions, _ = timeline.predict(img_emb)
        return {'predicted_year': int(predictions[0])}

    elif method == 'bezier':
        timeline = BezierTimeline(num_control_points=num_control_points)
        rdim = reduce_dim if reduce_dim and reduce_dim > 0 else None
        timeline.fit(timeline_emb, timeline_years, reduce_dim=rdim)
        if bezier_method == 'interpolation':
            preds = timeline.predict_interpolation(img_emb)
        else:
            preds = timeline.predict_nearest_neighbor(img_emb)
        return {'predicted_year': int(preds[0])}

    else:
        raise ValueError(f"Unknown method: {method}")


def process_image(image_path, model, preprocess, tokenizer, model_name,
                  method, device, prompt_template, years, precomputed,
                  bezier_method, reduce_dim, num_control_points):
    """Process a single image and return prediction."""
    image = Image.open(image_path).convert('RGB')

    t0 = time.perf_counter()

    if method == 'time_probing':
        result = predict_time_probing(
            model, preprocess, tokenizer, model_name, image,
            prompt_template, years, device
        )
    else:
        if precomputed is None:
            raise ValueError(
                f"Method '{method}' requires precomputed embeddings. "
                f"Use --embeddings_path or use --method time_probing"
            )
        result = predict_with_embeddings(
            model, preprocess, image, precomputed, method, device,
            bezier_method, reduce_dim, num_control_points
        )

    result['inference_ms'] = (time.perf_counter() - t0) * 1000
    result['image'] = str(image_path)
    result['model'] = model_name
    result['method'] = method

    return result


def format_output(result, output_format='text'):
    """Format prediction result for display."""
    if output_format == 'json':
        # Remove large arrays for clean JSON
        clean = {k: v for k, v in result.items() if k != 'all_similarities'}
        return json.dumps(clean, indent=2)

    lines = [
        f"Image:          {result['image']}",
        f"Predicted year: {result['predicted_year']}",
        f"Model:          {result['model']}",
        f"Method:         {result['method']}",
        f"Inference:      {result.get('inference_ms', 0):.0f}ms",
    ]

    if 'confidence_scores' in result:
        lines.append("Top predictions:")
        for year, score in result['confidence_scores'].items():
            lines.append(f"  {year}: {score:.2f}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Predict year of first appearance for images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --image photo.jpg
  python predict.py --image photo.jpg --model eva-clip-l14-336 --method bezier
  python predict.py --image_dir photos/ --method time_probing --prompt P7
  python predict.py --image photo.jpg --output json
        """
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to a single image')
    input_group.add_argument('--image_dir', type=str,
                             help='Directory of images to predict')

    # Model
    parser.add_argument('--model', type=str, default='clip-vit-b32',
                        help='VLM model name (default: clip-vit-b32)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'])

    # Method
    parser.add_argument('--method', type=str, default='bezier',
                        choices=['time_probing', 'umap', 'bezier'],
                        help='Prediction method (default: bezier)')

    # Time probing options
    parser.add_argument('--prompt', type=str, default='P7',
                        help='Prompt template ID for time probing (default: P7)')
    parser.add_argument('--year_min', type=int, default=1700)
    parser.add_argument('--year_max', type=int, default=2024)

    # Bézier options
    parser.add_argument('--reduce_dim', type=int, default=13,
                        help='KPCA dimension for Bézier R^S (0=no reduction)')
    parser.add_argument('--bezier_method', type=str, default='interpolation',
                        choices=['interpolation', 'nearest_neighbor'])
    parser.add_argument('--num_control_points', type=int, default=200)

    # Embeddings
    parser.add_argument('--embeddings_path', type=str, default='encodings',
                        help='Path to precomputed embeddings (for umap/bezier)')

    # Output
    parser.add_argument('--output', type=str, default='text',
                        choices=['text', 'json', 'csv'],
                        help='Output format')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to file')

    args = parser.parse_args()

    # Load model
    print(f"Loading {args.model}...", flush=True)
    model, preprocess, tokenizer = load_model(args.model, args.device)

    # Setup
    prompts = get_prompt_templates()
    prompt_template = prompts[args.prompt]
    years = list(range(args.year_min, args.year_max + 1))

    # Load precomputed embeddings if needed
    precomputed = None
    if args.method in ('umap', 'bezier'):
        if args.embeddings_path and os.path.exists(args.embeddings_path):
            from evaluation.embeddings import load_precomputed_embeddings
            try:
                precomputed = load_precomputed_embeddings(
                    args.embeddings_path, args.model
                )
                print(f"Loaded precomputed embeddings from {args.embeddings_path}")
            except FileNotFoundError:
                print(f"No precomputed embeddings for {args.model}, "
                      f"falling back to time_probing")
                args.method = 'time_probing'

    # Collect images
    if args.image:
        image_paths = [args.image]
    else:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_paths = sorted(
            str(p) for p in Path(args.image_dir).rglob('*')
            if p.suffix.lower() in exts
        )
        print(f"Found {len(image_paths)} images in {args.image_dir}")

    # Predict
    results = []
    for path in image_paths:
        try:
            result = process_image(
                path, model, preprocess, tokenizer, args.model,
                args.method, args.device, prompt_template, years,
                precomputed, args.bezier_method, args.reduce_dim,
                args.num_control_points,
            )
            results.append(result)

            if args.output == 'text':
                print(f"\n{format_output(result, 'text')}")

        except Exception as e:
            print(f"Error processing {path}: {e}")

    # Output
    if args.output == 'json':
        clean_results = [
            {k: v for k, v in r.items() if k != 'all_similarities'}
            for r in results
        ]
        print(json.dumps(clean_results, indent=2))

    elif args.output == 'csv':
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['image', 'predicted_year', 'model', 'method', 'inference_ms'])
        for r in results:
            writer.writerow([
                r['image'], r['predicted_year'], r['model'],
                r['method'], f"{r.get('inference_ms', 0):.0f}"
            ])
        print(output.getvalue())

    # Save
    if args.save:
        clean_results = [
            {k: v for k, v in r.items() if k != 'all_similarities'}
            for r in results
        ]
        with open(args.save, 'w') as f:
            json.dump(clean_results, f, indent=2)
        print(f"\nResults saved to {args.save}")


if __name__ == '__main__':
    main()
