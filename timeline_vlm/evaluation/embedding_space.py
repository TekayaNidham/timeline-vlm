"""
Embedding Space Analysis (Section 3.2 / Section 6.2).

Analyzes the spatial structure of time embeddings using:
- Kernel PCA (KPCA) with cosine kernel for dimensionality reduction
- 1D, 2D, 3D projections for visualization
- Chronological ordering quality assessment (Table 4)
- Optimal dimension analysis (Figure 6: MAE per dimension)
"""

import os
import argparse
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.decomposition import KernelPCA
import umap

from ..utils.metrics import compute_ranking_metrics, mean_absolute_error, calculate_TAI
from .timeline_bezier import BezierTimeline


def analyze_1d_progression(time_embeddings, years, method='kpca', model_name=None):
    """
    Analyze degree of chronological progression in 1D (Table 4).

    Reduces time embeddings to 1D using KPCA or UMAP, then evaluates
    ranking metrics.

    Args:
        time_embeddings: (T, D) time embeddings
        years: List of T years
        method: 'kpca' or 'umap'
        model_name: Model name (for UMAP parameter selection)

    Returns:
        dict with spearman_rho, kendall_tau, delta_MNDL
    """
    years_arr = np.array(years)

    if method == 'kpca':
        reducer = KernelPCA(n_components=1, kernel='cosine')
        projection_1d = reducer.fit_transform(time_embeddings).flatten()
    elif method == 'umap':
        from .timeline_umap import PAPER_PARAMS
        if model_name and 'eva' in model_name.lower():
            params = PAPER_PARAMS['eva']
        else:
            params = PAPER_PARAMS['clip']

        reducer = umap.UMAP(
            n_components=1,
            n_neighbors=params['n_neighbors'],
            min_dist=params['min_dist'],
            metric='cosine',
            random_state=42,
        )
        projection_1d = reducer.fit_transform(time_embeddings).flatten()
    else:
        raise ValueError(f"Unknown method: {method}")

    metrics = compute_ranking_metrics(projection_1d, years_arr)
    return metrics


def generate_table4(time_embeddings_clip, years_clip,
                    time_embeddings_eva, years_eva):
    """
    Reproduce Table 4: Degree of chronological progression in 1D.

    Evaluates both KPCA and UMAP on CLIP and EVA-CLIP embeddings.

    Returns:
        dict mapping (model, method) -> metrics
    """
    results = {}
    configs = [
        ('CLIP', time_embeddings_clip, years_clip, 'clip-vit-b32'),
        ('EVA-CLIP', time_embeddings_eva, years_eva, 'eva-clip-l14-336'),
    ]

    print("\nTable 4: Degree of chronological progression in 1D")
    print(f"{'Metric':<12} {'CLIP KPCA':>10} {'CLIP UMAP':>10} "
          f"{'EVA KPCA':>10} {'EVA UMAP':>10}")
    print('-' * 55)

    for method in ['kpca', 'umap']:
        for name, emb, yrs, model_name in configs:
            key = (name, method)
            results[key] = analyze_1d_progression(
                emb, yrs, method=method, model_name=model_name
            )

    for metric_key, metric_name in [('spearman_rho', 'ρ'),
                                     ('kendall_tau', 'τ'),
                                     ('delta_MNDL', 'δ_MNDL')]:
        vals = []
        for name in ['CLIP', 'EVA-CLIP']:
            for method in ['kpca', 'umap']:
                vals.append(results[(name, method)][metric_key])
        print(f"{metric_name:<12} {vals[0]:>10.2f} {vals[1]:>10.2f} "
              f"{vals[2]:>10.2f} {vals[3]:>10.2f}")

    return results


def analyze_dimension_sweep(time_embeddings, time_years,
                            image_embeddings, image_years,
                            max_dim=50, num_control_points=200):
    """
    Reproduce Figure 6: MAE per dimension for KPCA.

    Sweeps KPCA dimensions from 1 to max_dim and evaluates Bézier
    curve prediction accuracy at each dimension.

    Args:
        time_embeddings: (T, D) time embeddings
        time_years: List of T years
        image_embeddings: (N, D) image embeddings
        image_years: (N,) true years
        max_dim: Maximum dimension to test
        num_control_points: Bézier K parameter

    Returns:
        dict mapping dimension -> {mae, tai}
    """
    image_years = np.array(image_years)
    results = {}
    max_possible = min(max_dim, time_embeddings.shape[1],
                       time_embeddings.shape[0] - 1)

    print(f"Dimension sweep (1 to {max_possible})...")
    for dim in range(1, max_possible + 1):
        bezier = BezierTimeline(num_control_points=num_control_points)
        bezier.fit(time_embeddings, time_years, reduce_dim=dim)
        res = bezier.evaluate(image_embeddings, image_years,
                              method='interpolation')
        results[dim] = {'mae': res['mae'], 'tai': res['tai']}

        if dim <= 20 or dim % 5 == 0:
            print(f"  dim={dim:>3}: MAE={res['mae']:.2f}, TAI={res['tai']:.3f}")

    return results


def plot_dimension_sweep(results_clip, results_eva=None, save_path=None):
    """
    Plot Figure 6: MAE per dimension for (EVA-)CLIP using KPCA.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    dims = sorted(results_clip.keys())
    maes_clip = [results_clip[d]['mae'] for d in dims]
    ax.plot(dims, maes_clip, 'o-', label='CLIP', markersize=3)

    if results_eva:
        dims_eva = sorted(results_eva.keys())
        maes_eva = [results_eva[d]['mae'] for d in dims_eva]
        ax.plot(dims_eva, maes_eva, 's-', label='EVA-CLIP', markersize=3)

    ax.set_xlabel('Dimension')
    ax.set_ylabel('MAE')
    ax.set_title('MAE per dimension for (EVA-)CLIP using KPCA')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark optimal dimension (S=13)
    ax.axvline(x=13, color='gray', linestyle='--', alpha=0.5, label='S=13')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Embedding Space Analysis (Table 4 & Figure 6)')
    parser.add_argument('--embeddings_path', type=str, default='encodings')
    parser.add_argument('--table4', action='store_true',
                        help='Generate Table 4 results')
    parser.add_argument('--figure6', action='store_true',
                        help='Generate Figure 6 dimension sweep')
    parser.add_argument('--max_dim', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='results/')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from .embeddings import load_precomputed_embeddings

    if args.table4:
        clip_data = load_precomputed_embeddings(args.embeddings_path, 'clip-vit-b32')
        eva_data = load_precomputed_embeddings(args.embeddings_path, 'eva-clip-l14-336')
        generate_table4(
            clip_data['timeline_emb'], clip_data['timeline_years'],
            eva_data['timeline_emb'], eva_data['timeline_years'],
        )

    if args.figure6:
        for model_name, label in [('clip-vit-b32', 'CLIP'),
                                   ('eva-clip-l14-336', 'EVA-CLIP')]:
            try:
                data = load_precomputed_embeddings(
                    args.embeddings_path, model_name
                )
                print(f"\n{label} dimension sweep:")
                results = analyze_dimension_sweep(
                    data['timeline_emb'], data['timeline_years'],
                    data['image_emb'], data['image_years'],
                    max_dim=args.max_dim,
                )

                import json
                out_path = os.path.join(
                    args.output_dir, f'dimension_sweep_{label.lower()}.json'
                )
                with open(out_path, 'w') as f:
                    json.dump({str(k): v for k, v in results.items()}, f, indent=2)
                print(f"Saved to {out_path}")

            except FileNotFoundError as e:
                print(f"Skipping {label}: {e}")


if __name__ == '__main__':
    main()
