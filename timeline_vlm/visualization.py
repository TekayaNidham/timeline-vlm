"""
Visualization utilities for timeline-vlm.

Provides functions to visualize predictions on the temporal timeline,
embedding manifolds, and Bezier curves. Supports 1D, 2D, and 3D projections.
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


def plot_prediction(predictor, image_path, dim=3, save_path=None, show=True):
    """
    Visualize a prediction on the KPCA timeline.

    Shows the input image alongside the embedding space with the
    predicted position marked on the timeline.

    Args:
        predictor: A fitted TimelinePredictor instance.
        image_path: Path to the image to predict.
        dim: Projection dimensionality (1, 2, or 3). Default: 3.
        save_path: If set, save the figure to this path.
        show: If True, display the plot interactively.

    Returns:
        dict with predicted_year and figure object.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import KernelPCA
    from PIL import Image

    if dim not in (1, 2, 3):
        raise ValueError(f"dim must be 1, 2, or 3, got {dim}")

    if not predictor.is_fitted:
        raise ValueError("Predictor must be fitted before visualization.")

    predictor._ensure_model()

    img_emb = predictor._encode_images([image_path])
    pred_year = int(predictor._predict_from_embeddings(img_emb)[0])

    image = Image.open(image_path).convert('RGB')

    data = _load_timeline_embeddings(predictor)
    years = np.array(data['timeline_years'])
    year_norm = (years - years.min()) / (years.max() - years.min())

    kpca = KernelPCA(n_components=dim, kernel='cosine')
    time_proj = kpca.fit_transform(data['timeline_emb'])
    img_proj = kpca.transform(img_emb)

    if dim == 1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                                 gridspec_kw={'width_ratios': [1, 2]})
        axes[0].imshow(image)
        axes[0].set_title(f'Predicted: {pred_year}', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        sc = axes[1].scatter(time_proj[:, 0], years,
                             c=year_norm, cmap='viridis', s=20, alpha=0.5)
        axes[1].scatter(img_proj[:, 0], [pred_year],
                        c='red', s=200, marker='*', zorder=5,
                        label=f'Prediction: {pred_year}')
        axes[1].set_xlabel('KPCA PC1')
        axes[1].set_ylabel('Year')
        axes[1].legend(fontsize=12)
        axes[1].set_title(f'{predictor.model_name} — 1D KPCA Timeline')
        axes[1].grid(True, alpha=0.3)
        cbar = plt.colorbar(sc, ax=axes[1])
        cbar.set_label('Year')
        cbar.set_ticks(np.linspace(0, 1, 7))
        cbar.set_ticklabels([str(int(y)) for y in np.linspace(years.min(), years.max(), 7)])

    elif dim == 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                                 gridspec_kw={'width_ratios': [1, 2]})
        axes[0].imshow(image)
        axes[0].set_title(f'Predicted: {pred_year}', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        sc = axes[1].scatter(time_proj[:, 0], time_proj[:, 1],
                             c=year_norm, cmap='viridis', s=15, alpha=0.5)
        axes[1].scatter(img_proj[:, 0], img_proj[:, 1],
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

    else:  # dim == 3
        fig = plt.figure(figsize=(18, 7))
        ax_img = fig.add_subplot(1, 2, 1)
        ax_3d = fig.add_subplot(1, 2, 2, projection='3d')

        ax_img.imshow(image)
        ax_img.set_title(f'Predicted: {pred_year}', fontsize=14, fontweight='bold')
        ax_img.axis('off')

        sc = ax_3d.scatter(time_proj[:, 0], time_proj[:, 1], time_proj[:, 2],
                           c=year_norm, cmap='viridis', s=15, alpha=0.5)
        ax_3d.scatter(img_proj[:, 0], img_proj[:, 1], img_proj[:, 2],
                      c='red', s=200, marker='*', zorder=5,
                      label=f'Prediction: {pred_year}')
        ax_3d.legend(fontsize=12)
        ax_3d.set_xlabel('PC1')
        ax_3d.set_ylabel('PC2')
        ax_3d.set_zlabel('PC3')
        ax_3d.set_title(f'{predictor.model_name} — 3D KPCA Timeline')

        cbar = fig.colorbar(sc, ax=ax_3d, shrink=0.6, pad=0.1)
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


def plot_timeline(predictor, dim=3, save_path=None, show=True):
    """
    Visualize the fitted timeline as a KPCA projection colored by year.

    Args:
        predictor: A fitted TimelinePredictor instance.
        dim: Projection dimensionality (1, 2, or 3). Default: 3.
        save_path: If set, save the figure to this path.
        show: If True, display the plot interactively.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import KernelPCA

    if dim not in (1, 2, 3):
        raise ValueError(f"dim must be 1, 2, or 3, got {dim}")

    if not predictor.is_fitted:
        raise ValueError("Predictor must be fitted before visualization.")

    data = _load_timeline_embeddings(predictor)
    years = np.array(data['timeline_years'])
    year_norm = (years - years.min()) / (years.max() - years.min())

    kpca = KernelPCA(n_components=dim, kernel='cosine')
    proj = kpca.fit_transform(data['timeline_emb'])

    if dim == 1:
        fig, ax = plt.subplots(figsize=(14, 5))
        sc = ax.scatter(proj[:, 0], years, c=years, cmap='viridis', s=20, alpha=0.7)
        ax.set_xlabel('KPCA 1D Projection')
        ax.set_ylabel('Year')
        ax.set_title(f'{predictor.model_name} — 1D Temporal Timeline')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Year')

    elif dim == 2:
        fig, ax = plt.subplots(figsize=(14, 8))
        sc = ax.scatter(proj[:, 0], proj[:, 1],
                        c=year_norm, cmap='viridis', s=20, alpha=0.7)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'{predictor.model_name} — 2D Temporal Timeline')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Year')
        cbar.set_ticks(np.linspace(0, 1, 7))
        cbar.set_ticklabels([str(int(y)) for y in np.linspace(years.min(), years.max(), 7)])

    else:  # dim == 3
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                        c=year_norm, cmap='viridis', s=20, alpha=0.7)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'{predictor.model_name} — 3D Temporal Timeline')
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
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

    return fig
