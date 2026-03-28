"""
Visualization utilities for timeline-vlm.

Provides functions to visualize predictions on the temporal timeline,
embedding manifolds, and Bezier curves.
"""

import numpy as np


def _load_timeline_embeddings(predictor):
    """Load timeline embeddings, using the predictor's stored path if available."""
    from .evaluation.embeddings import load_precomputed_embeddings

    paths = []
    if predictor._embeddings_path:
        paths.append(predictor._embeddings_path)
    paths.extend(['encodings', 'data/encodings'])

    for emb_path in paths:
        try:
            return load_precomputed_embeddings(emb_path, predictor.model_name)
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        "Cannot find precomputed embeddings for visualization. "
        "Pass embeddings_path to fit_from_precomputed()."
    )


def plot_prediction(predictor, image_path, save_path=None, show=True):
    """
    Visualize a prediction on the 2D KPCA timeline.

    Shows the input image alongside the embedding space with the
    predicted position marked on the timeline.

    Args:
        predictor: A fitted TimelinePredictor instance.
        image_path: Path to the image to predict.
        save_path: If set, save the figure to this path.
        show: If True, display the plot interactively.

    Returns:
        dict with predicted_year and figure object.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import KernelPCA
    from PIL import Image

    if not predictor.is_fitted:
        raise ValueError("Predictor must be fitted before visualization.")

    predictor._ensure_model()

    # Encode via the predictor's own method (consistent preprocessing)
    img_emb = predictor._encode_images([image_path])
    pred_year = int(predictor._predict_from_embeddings(img_emb)[0])

    # Load image for display
    image = Image.open(image_path).convert('RGB')

    data = _load_timeline_embeddings(predictor)
    years = np.array(data['timeline_years'])
    year_norm = (years - years.min()) / (years.max() - years.min())

    # Project to 2D
    kpca = KernelPCA(n_components=2, kernel='cosine')
    time_2d = kpca.fit_transform(data['timeline_emb'])
    img_2d = kpca.transform(img_emb)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={'width_ratios': [1, 2]})

    # Show image
    axes[0].imshow(image)
    axes[0].set_title(f'Predicted: {pred_year}', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Timeline with prediction
    sc = axes[1].scatter(time_2d[:, 0], time_2d[:, 1],
                         c=year_norm, cmap='viridis', s=15, alpha=0.5)
    axes[1].scatter(img_2d[:, 0], img_2d[:, 1],
                    c='red', s=200, marker='*', zorder=5,
                    label=f'Prediction: {pred_year}')
    axes[1].legend(fontsize=12)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title(f'{predictor.model_name} — 2D KPCA Timeline')
    axes[1].grid(True, alpha=0.3)

    cbar = plt.colorbar(sc, ax=axes[1])
    cbar.set_label('Year')
    cbar.set_ticks(np.linspace(0, 1, 7))
    cbar.set_ticklabels([str(int(y)) for y in np.linspace(years.min(), years.max(), 7)])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {'predicted_year': pred_year, 'figure': fig}


def plot_timeline(predictor, save_path=None, show=True):
    """
    Visualize the fitted timeline as a 1D projection colored by year.

    Args:
        predictor: A fitted TimelinePredictor instance.
        save_path: If set, save the figure to this path.
        show: If True, display the plot interactively.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import KernelPCA

    if not predictor.is_fitted:
        raise ValueError("Predictor must be fitted before visualization.")

    data = _load_timeline_embeddings(predictor)
    years = np.array(data['timeline_years'])

    # 1D KPCA projection
    kpca = KernelPCA(n_components=1, kernel='cosine')
    proj_1d = kpca.fit_transform(data['timeline_emb']).flatten()

    fig, ax = plt.subplots(figsize=(14, 5))
    sc = ax.scatter(proj_1d, years, c=years, cmap='viridis', s=20, alpha=0.7)
    ax.set_xlabel('KPCA 1D Projection')
    ax.set_ylabel('Year')
    ax.set_title(f'{predictor.model_name} — Temporal Timeline')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Year')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
