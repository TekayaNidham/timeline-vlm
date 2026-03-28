"""
Command-line interface for timeline-vlm.

Installed as `timeline-vlm` console script via pip.

Usage:
    timeline-vlm predict photo.jpg
    timeline-vlm predict photo.jpg --model "CLIP ViT-L/14" --method time_probing
    timeline-vlm predict photos/ --output json --save results.json
    timeline-vlm list-models
    timeline-vlm visualize timeline --model clip-vit-b32 --save timeline.png
"""

import argparse
import json
import sys
import time
from pathlib import Path


def cmd_predict(args):
    """Predict year of first appearance for images."""
    from .predictor import TimelinePredictor

    predictor = TimelinePredictor(
        model=args.model,
        method=args.method,
        device=args.device,
        prompt=args.prompt,
        reduce_dim=args.reduce_dim,
        bezier_method=args.bezier_method,
    )

    if args.embeddings_path:
        predictor.fit_from_precomputed(args.embeddings_path)
    else:
        predictor.fit_from_dataset()

    # Collect image paths
    target = Path(args.target)
    if target.is_dir():
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_paths = sorted(str(p) for p in target.rglob('*') if p.suffix.lower() in exts)
        print(f"Found {len(image_paths)} images in {target}")
    elif target.is_file():
        image_paths = [str(target)]
    else:
        print(f"Error: '{args.target}' is not a valid file or directory.", file=sys.stderr)
        sys.exit(1)

    results = []
    for path in image_paths:
        try:
            details = predictor.predict_with_details(path)
            result = {
                'image': path,
                'predicted_year': details['predicted_year'],
                'model': details['model'],
                'method': details['method'],
                'inference_ms': round(details['inference_ms'], 1),
            }

            if 'top_predictions' in details:
                result['top_predictions'] = details['top_predictions']

            results.append(result)

            if args.output == 'text':
                print(f"{path} -> {details['predicted_year']}")

        except Exception as e:
            print(f"Error: {path}: {e}", file=sys.stderr)

    if args.output == 'json':
        print(json.dumps(results, indent=2))

    if args.save:
        with open(args.save, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.save}")


def cmd_list_models(args):
    """List all supported models."""
    from . import list_models
    from .models.model_loader import MODEL_DISPLAY_NAMES, TIMELINE_SUPPORTED_MODELS

    models = list_models(verbose=True)

    timeline_names = sorted(
        MODEL_DISPLAY_NAMES[k] for k in TIMELINE_SUPPORTED_MODELS
    )
    time_probing_names = sorted(models.keys())

    if args.verbose:
        print(f"\n{'Display Name':<45} {'Key':<35} {'Methods'}")
        print('-' * 110)
        for name, info in sorted(models.items()):
            methods = ', '.join(info['methods'])
            print(f"{name:<45} {info['key']:<35} {methods}")
    else:
        print(f"\nTimeline prediction (Bezier/UMAP) — {len(timeline_names)} models:")
        for name in timeline_names:
            print(f"  {name}")
        print(f"  (more models coming soon)\n")
        print(f"Time probing — all {len(time_probing_names)} models:")
        for name in time_probing_names:
            print(f"  {name}")


def cmd_visualize(args):
    """Visualize timelines and predictions."""
    from .predictor import TimelinePredictor

    predictor = TimelinePredictor(model=args.model, device=args.device)
    predictor.fit_from_precomputed(args.embeddings_path)

    if args.type == 'prediction' and args.image:
        from .visualization import plot_prediction
        plot_prediction(predictor, args.image, dim=args.dim,
                        save_path=args.save, show=not args.save)
    elif args.type == 'timeline':
        from .visualization import plot_timeline
        plot_timeline(predictor, dim=args.dim,
                      save_path=args.save, show=not args.save)
    else:
        print("Specify --type (timeline or prediction). "
              "For prediction, also provide --image.")


def main():
    parser = argparse.ArgumentParser(
        prog='timeline-vlm',
        description='Temporal inference with Vision-Language Models',
    )
    subparsers = parser.add_subparsers(dest='command')

    # predict
    p = subparsers.add_parser('predict', help='Predict year of an image')
    p.add_argument('target', help='Image file or directory')
    p.add_argument('--model', default='clip-vit-b32',
                   help='Model name or display name (default: clip-vit-b32)')
    p.add_argument('--method', default='bezier',
                   choices=['time_probing', 'umap', 'bezier'])
    p.add_argument('--device', default='cpu', choices=['cuda', 'cpu'])
    p.add_argument('--embeddings_path', default='encodings')
    p.add_argument('--prompt', default='P7')
    p.add_argument('--reduce_dim', type=int, default=None,
                   help='KPCA dimensions for Bezier (default: None = original space)')
    p.add_argument('--bezier_method', default='interpolation',
                   choices=['interpolation', 'nearest_neighbor'])
    p.add_argument('--output', default='text', choices=['text', 'json'])
    p.add_argument('--save', default=None, help='Save results to JSON file')

    # list-models
    p = subparsers.add_parser('list-models', help='List supported models')
    p.add_argument('--verbose', '-v', action='store_true')

    # visualize
    p = subparsers.add_parser('visualize', help='Visualize timelines')
    p.add_argument('type', choices=['timeline', 'prediction'])
    p.add_argument('--model', default='clip-vit-b32')
    p.add_argument('--device', default='cpu')
    p.add_argument('--embeddings_path', default='encodings')
    p.add_argument('--dim', type=int, default=3, choices=[1, 2, 3],
                   help='Projection dimensionality (default: 3)')
    p.add_argument('--image', default=None, help='Image for prediction viz')
    p.add_argument('--save', default=None)

    args = parser.parse_args()

    if args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'list-models':
        cmd_list_models(args)
    elif args.command == 'visualize':
        cmd_visualize(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
