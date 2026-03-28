"""
Visualize temporal structure in VLM embedding spaces.

Interactive exploration of timelines, embedding manifolds, and predictions.

Usage:
    # 3D manifold visualization (KPCA)
    python visualize.py manifold --model clip-vit-b32 --dim 3

    # 1D timeline projection comparison (KPCA vs UMAP)
    python visualize.py timeline --model clip-vit-b32

    # Bézier curve visualization
    python visualize.py bezier --model clip-vit-b32 --reduce_dim 13

    # Dimension sweep (Figure 6)
    python visualize.py dimension_sweep --model clip-vit-b32 --max_dim 50

    # Prediction visualization for images
    python visualize.py predict --image photo.jpg --model clip-vit-b32

    # Year distribution heatmap
    python visualize.py distribution --embeddings_path encodings
"""

import argparse
import numpy as np
from pathlib import Path


def cmd_manifold(args):
    """Visualize 2D/3D embedding manifold via KPCA."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import KernelPCA
    from timeline_vlm.evaluation.embeddings import load_precomputed_embeddings

    data = load_precomputed_embeddings(args.embeddings_path, args.model)
    dim = args.dim

    print(f"Projecting to {dim}D via KPCA (cosine)...")
    kpca = KernelPCA(n_components=dim, kernel='cosine')
    time_proj = kpca.fit_transform(data['timeline_emb'])

    years = np.array(data['timeline_years'])
    year_norm = (years - years.min()) / (years.max() - years.min())
    colors = plt.cm.viridis(year_norm)

    if dim == 3:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(time_proj[:, 0], time_proj[:, 1], time_proj[:, 2],
                        c=year_norm, cmap='viridis', s=25, alpha=0.8)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

        # Add year labels at intervals
        for i in range(0, len(years), max(1, len(years) // 15)):
            ax.text(time_proj[i, 0], time_proj[i, 1], time_proj[i, 2],
                    str(years[i]), fontsize=7, alpha=0.7)

        # Optionally show image embeddings
        if args.show_images:
            img_proj = kpca.transform(data['image_emb'])
            ax.scatter(img_proj[:, 0], img_proj[:, 1], img_proj[:, 2],
                       c='black', s=3, alpha=0.1, marker='.')

    elif dim == 2:
        fig, ax = plt.subplots(figsize=(12, 8))
        sc = ax.scatter(time_proj[:, 0], time_proj[:, 1],
                        c=year_norm, cmap='viridis', s=30, alpha=0.8)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        for i in range(0, len(years), max(1, len(years) // 20)):
            ax.annotate(str(years[i]), (time_proj[i, 0], time_proj[i, 1]),
                        fontsize=7, alpha=0.6)
    else:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.scatter(time_proj[:, 0], np.zeros(len(time_proj)),
                   c=year_norm, cmap='viridis', s=30)
        ax.set_xlabel('PC1')
        ax.set_yticks([])

    cbar = plt.colorbar(sc if dim > 1 else
                        plt.cm.ScalarMappable(cmap='viridis'), ax=ax)
    cbar.set_label('Year')
    cbar.set_ticks(np.linspace(0, 1, 7))
    cbar.set_ticklabels([str(int(y)) for y in
                         np.linspace(years.min(), years.max(), 7)])

    ax.set_title(f'{args.model} — {dim}D KPCA Embedding Manifold')
    plt.tight_layout()
    _save_or_show(fig, args.save)


def cmd_timeline(args):
    """Compare 1D timeline projections: KPCA vs UMAP."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import KernelPCA
    import umap
    from timeline_vlm.evaluation.embeddings import load_precomputed_embeddings
    from timeline_vlm.evaluation.timeline_umap import PAPER_PARAMS

    data = load_precomputed_embeddings(args.embeddings_path, args.model)
    years = np.array(data['timeline_years'])

    # KPCA 1D
    kpca = KernelPCA(n_components=1, kernel='cosine')
    kpca_1d = kpca.fit_transform(data['timeline_emb']).flatten()

    # UMAP 1D
    params = PAPER_PARAMS.get('eva' if 'eva' in args.model.lower() else 'clip')
    reducer = umap.UMAP(
        n_components=1, n_neighbors=params['n_neighbors'],
        min_dist=params['min_dist'], metric='cosine', random_state=42,
    )
    umap_1d = reducer.fit_transform(data['timeline_emb']).flatten()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # KPCA timeline
    axes[0].scatter(kpca_1d, years, c=years, cmap='viridis', s=15, alpha=0.7)
    axes[0].set_ylabel('Year')
    axes[0].set_xlabel('KPCA 1D Projection')
    axes[0].set_title(f'{args.model} — KPCA Timeline')
    axes[0].grid(True, alpha=0.3)

    # UMAP timeline
    axes[1].scatter(umap_1d, years, c=years, cmap='viridis', s=15, alpha=0.7)
    axes[1].set_ylabel('Year')
    axes[1].set_xlabel('UMAP 1D Projection')
    axes[1].set_title(f'{args.model} — UMAP Timeline')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, args.save)


def cmd_bezier(args):
    """Visualize Bézier curve timeline in 3D."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import KernelPCA
    from timeline_vlm.evaluation.timeline_bezier import BezierTimeline
    from timeline_vlm.evaluation.embeddings import load_precomputed_embeddings

    data = load_precomputed_embeddings(args.embeddings_path, args.model)
    years = np.array(data['timeline_years'])

    # Project to 3D for visualization
    kpca_3d = KernelPCA(n_components=3, kernel='cosine')
    time_3d = kpca_3d.fit_transform(data['timeline_emb'])

    # Fit Bézier in 3D
    bezier = BezierTimeline(num_control_points=args.num_control_points)
    K = min(args.num_control_points, len(time_3d))
    indices = np.linspace(0, len(time_3d) - 1, K, dtype=int)
    cp_3d = time_3d[indices]
    curve_3d = bezier.bezier_curve(cp_3d)

    # Optionally fit in reduce_dim space too
    if args.reduce_dim and args.reduce_dim > 0:
        bezier_fit = BezierTimeline(num_control_points=args.num_control_points)
        bezier_fit.fit(data['timeline_emb'], data['timeline_years'],
                       reduce_dim=args.reduce_dim)

    year_norm = (years - years.min()) / (years.max() - years.min())

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Time embeddings
    ax.scatter(time_3d[:, 0], time_3d[:, 1], time_3d[:, 2],
               c=year_norm, cmap='viridis', s=20, alpha=0.6,
               label='Time embeddings')

    # Bézier curve
    ax.plot(curve_3d[:, 0], curve_3d[:, 1], curve_3d[:, 2],
            'r-', linewidth=2, label='Bézier curve')

    # Control points
    ax.scatter(cp_3d[::5, 0], cp_3d[::5, 1], cp_3d[::5, 2],
               c='red', s=40, marker='D', alpha=0.5, label='Control points')

    # Image embeddings
    if args.show_images:
        img_3d = kpca_3d.transform(data['image_emb'])
        ax.scatter(img_3d[:, 0], img_3d[:, 1], img_3d[:, 2],
                   c='black', s=3, alpha=0.05, marker='.')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()
    ax.set_title(f'{args.model} — Bézier Timeline (K={K})')

    plt.tight_layout()
    _save_or_show(fig, args.save)


def cmd_dimension_sweep(args):
    """Plot MAE per KPCA dimension (Figure 6)."""
    import matplotlib.pyplot as plt
    from timeline_vlm.evaluation.embedding_space import analyze_dimension_sweep
    from timeline_vlm.evaluation.embeddings import load_precomputed_embeddings

    models = args.models or [args.model]
    all_results = {}

    for model_name in models:
        print(f"\n{model_name}:")
        try:
            data = load_precomputed_embeddings(args.embeddings_path, model_name)
            results = analyze_dimension_sweep(
                data['timeline_emb'], data['timeline_years'],
                data['image_emb'], data['image_years'],
                max_dim=args.max_dim,
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"  Error: {e}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, results in all_results.items():
        dims = sorted(results.keys())
        maes = [results[d]['mae'] for d in dims]
        label = 'CLIP' if 'clip-vit' in model_name else 'EVA-CLIP'
        ax.plot(dims, maes, 'o-', label=label, markersize=4)

    ax.axvline(x=13, color='gray', linestyle='--', alpha=0.5)
    ax.annotate('S=13 (optimal)', xy=(13, ax.get_ylim()[0]),
                fontsize=9, alpha=0.6)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('MAE (years)')
    ax.set_title('MAE per KPCA Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, args.save)


def cmd_distribution(args):
    """Visualize year distribution of predictions vs ground truth."""
    import matplotlib.pyplot as plt
    from timeline_vlm.evaluation.embeddings import load_precomputed_embeddings
    from timeline_vlm.evaluation.timeline_bezier import BezierTimeline

    data = load_precomputed_embeddings(args.embeddings_path, args.model)

    # Get predictions
    bezier = BezierTimeline(num_control_points=200)
    bezier.fit(data['timeline_emb'], data['timeline_years'], reduce_dim=13)
    preds = bezier.predict_interpolation(data['image_emb'])
    gt = data['image_years']

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Histogram comparison
    bins = np.arange(1700, 2030, 10)
    axes[0].hist(gt, bins=bins, alpha=0.5, label='Ground Truth', color='steelblue')
    axes[0].hist(preds, bins=bins, alpha=0.5, label='Predicted', color='coral')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Year Distribution')
    axes[0].legend()

    # Scatter: predicted vs actual
    axes[1].scatter(gt, preds, s=3, alpha=0.1, c='steelblue')
    axes[1].plot([1700, 2024], [1700, 2024], 'r--', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Ground Truth Year')
    axes[1].set_ylabel('Predicted Year')
    axes[1].set_title('Predicted vs. Ground Truth')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'{args.model} — Bézier(R^S, Int)', fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, args.save)


def cmd_predict(args):
    """Visualize prediction for a specific image on the timeline."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import KernelPCA
    from timeline_vlm.evaluation.timeline_bezier import BezierTimeline
    from timeline_vlm.evaluation.embeddings import load_precomputed_embeddings
    from timeline_vlm.models.model_loader import load_model
    from PIL import Image
    import torch

    data = load_precomputed_embeddings(args.embeddings_path, args.model)
    years = np.array(data['timeline_years'])

    # Load model and encode image
    model, preprocess, tokenizer = load_model(args.model, args.device)
    image = Image.open(args.image).convert('RGB')
    img_tensor = preprocess(image).unsqueeze(0).to(args.device)
    with torch.no_grad():
        img_emb = model.encode_image(img_tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        img_emb = img_emb.cpu().numpy()

    # Fit Bézier and predict
    bezier = BezierTimeline(num_control_points=200)
    bezier.fit(data['timeline_emb'], data['timeline_years'], reduce_dim=13)
    pred_year = int(bezier.predict_interpolation(img_emb)[0])

    # 2D visualization
    kpca = KernelPCA(n_components=2, kernel='cosine')
    time_2d = kpca.fit_transform(data['timeline_emb'])
    img_2d = kpca.transform(img_emb)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={'width_ratios': [1, 2]})

    # Image
    axes[0].imshow(image)
    axes[0].set_title(f'Predicted: {pred_year}', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Timeline with image projected
    year_norm = (years - years.min()) / (years.max() - years.min())
    axes[1].scatter(time_2d[:, 0], time_2d[:, 1],
                    c=year_norm, cmap='viridis', s=15, alpha=0.5)
    axes[1].scatter(img_2d[:, 0], img_2d[:, 1],
                    c='red', s=200, marker='*', zorder=5,
                    label=f'Image → {pred_year}')
    axes[1].legend(fontsize=12)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title(f'{args.model} — 2D KPCA Space')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, args.save)
    print(f"\nPredicted year: {pred_year}")


def _save_or_show(fig, save_path):
    """Save figure to file or display interactively."""
    import matplotlib.pyplot as plt
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize temporal structure in VLM embedding spaces',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Visualization type')

    # Shared args
    def add_common(p):
        p.add_argument('--model', type=str, default='clip-vit-b32')
        p.add_argument('--embeddings_path', type=str, default='encodings')
        p.add_argument('--save', type=str, default=None,
                       help='Save figure to file (e.g., output.png)')
        p.add_argument('--device', type=str, default='cpu')

    # Manifold
    p = subparsers.add_parser('manifold', help='2D/3D embedding manifold')
    add_common(p)
    p.add_argument('--dim', type=int, default=3, choices=[1, 2, 3])
    p.add_argument('--show_images', action='store_true',
                   help='Overlay image embeddings')

    # Timeline
    p = subparsers.add_parser('timeline', help='1D timeline (KPCA vs UMAP)')
    add_common(p)

    # Bézier
    p = subparsers.add_parser('bezier', help='Bézier curve in 3D')
    add_common(p)
    p.add_argument('--reduce_dim', type=int, default=13)
    p.add_argument('--num_control_points', type=int, default=200)
    p.add_argument('--show_images', action='store_true')

    # Dimension sweep
    p = subparsers.add_parser('dimension_sweep', help='MAE per dimension (Fig. 6)')
    add_common(p)
    p.add_argument('--max_dim', type=int, default=50)
    p.add_argument('--models', nargs='+', default=None,
                   help='Multiple models to compare')

    # Distribution
    p = subparsers.add_parser('distribution',
                              help='Prediction vs ground truth distributions')
    add_common(p)

    # Predict
    p = subparsers.add_parser('predict',
                              help='Visualize prediction for an image')
    add_common(p)
    p.add_argument('--image', type=str, required=True)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        'manifold': cmd_manifold,
        'timeline': cmd_timeline,
        'bezier': cmd_bezier,
        'dimension_sweep': cmd_dimension_sweep,
        'distribution': cmd_distribution,
        'predict': cmd_predict,
    }
    commands[args.command](args)


if __name__ == '__main__':
    main()
