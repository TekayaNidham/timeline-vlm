"""
UMAP-based Timeline Modeling (Section 3.3.1).

Projects time embeddings to a 1D manifold using UMAP, then maps image
embeddings onto the learned timeline for temporal inference.

Key parameters from paper (Section 5.3):
- CLIP (ViT-B/32): n_neighbors=38, min_dist=0.7446
- EVA-CLIP (EVA02-CLIP-L-14-336): n_neighbors=21, min_dist=0.1040
- Metric: cosine
- Optimization: TPE via Optuna, maximizing Spearman's rank correlation
"""

import argparse
import time
import numpy as np
from scipy.stats import spearmanr, kendalltau
import umap
from tabulate import tabulate

from ..utils.metrics import (calculate_TAI, mean_absolute_error,
                             evaluate_chronological_order, compute_ranking_metrics,
                             print_evaluation_summary)


# Paper-optimized parameters (Section 5.3)
PAPER_PARAMS = {
    'clip': {'n_neighbors': 38, 'min_dist': 0.7446},
    'eva':  {'n_neighbors': 21, 'min_dist': 0.1040},
}


class UMAPTimeline:
    """
    UMAP-based timeline for temporal inference.

    Learns a 1D projection of time embeddings that preserves
    chronological order, then maps image embeddings onto it.
    """

    def __init__(self, n_components=1, metric='cosine', random_state=42):
        self.n_components = n_components
        self.metric = metric
        self.random_state = random_state
        self.umap_model = None
        self.timeline_1d = None
        self.timeline_years = None

    def optimize_parameters(self, time_embeddings, years, n_trials=100):
        """
        Optimize UMAP hyperparameters via TPE (Optuna) to maximize
        Spearman's rank correlation between 1D projection and year order.
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            n_neighbors = trial.suggest_int('n_neighbors', 5, 50)
            min_dist = trial.suggest_float('min_dist', 0.0, 0.99)

            reducer = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=self.metric,
                random_state=self.random_state,
            )
            embedding_1d = reducer.fit_transform(time_embeddings).flatten()

            # Maximize absolute Spearman correlation
            rho, _ = spearmanr(embedding_1d, years)
            return -abs(rho)

        study = optuna.create_study()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        print(f"Optimized UMAP params: n_neighbors={best['n_neighbors']}, "
              f"min_dist={best['min_dist']:.4f}")
        return best

    def _get_default_params(self, model_name=None):
        """Get paper-optimized parameters based on model name."""
        if model_name and 'eva' in model_name.lower():
            return PAPER_PARAMS['eva'].copy()
        return PAPER_PARAMS['clip'].copy()

    def fit(self, time_embeddings, years, optimize=False, params=None,
            model_name=None, n_trials=100):
        """
        Fit UMAP model to time embeddings.

        Args:
            time_embeddings: (T, D) array of time embeddings
            years: List of T years (must be chronologically ordered)
            optimize: Run hyperparameter optimization
            params: Manual params dict {n_neighbors, min_dist}
            model_name: Model name for default param selection
            n_trials: Optuna trials if optimizing
        """
        self.timeline_years = list(years)

        if params is None:
            if optimize:
                params = self.optimize_parameters(
                    time_embeddings, years, n_trials
                )
            else:
                params = self._get_default_params(model_name)
                print(f"Using paper parameters: {params}")

        self.umap_model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=params['n_neighbors'],
            min_dist=params['min_dist'],
            metric=self.metric,
            random_state=self.random_state,
            transform_seed=self.random_state,
        )

        self.timeline_1d = self.umap_model.fit_transform(
            time_embeddings
        ).flatten()

        # Handle negative correlation (flip if needed for consistent ordering)
        rho, _ = spearmanr(self.timeline_1d, years)
        self._flip = rho < 0
        if self._flip:
            self.timeline_1d = -self.timeline_1d
            print(f"  Flipped timeline direction (original rho={rho:.3f})")

        # Evaluate quality
        quality = self._evaluate_quality()
        return quality

    def _evaluate_quality(self):
        """Evaluate chronological ordering quality of the timeline."""
        metrics = compute_ranking_metrics(
            self.timeline_1d,
            np.array(self.timeline_years)
        )

        print(f"\nTimeline quality:")
        print(f"  Spearman ρ:  {metrics['spearman_rho']:.3f}")
        print(f"  Kendall τ:   {metrics['kendall_tau']:.3f}")
        print(f"  δ_MNDL:      {metrics['delta_MNDL']:.3f}")

        return metrics

    def predict(self, image_embeddings):
        """
        Predict years for image embeddings.

        Maps images to 1D timeline via learned UMAP transform,
        then finds nearest year point.
        """
        if self.umap_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        t0 = time.perf_counter()

        # Transform to 1D
        image_1d = self.umap_model.transform(image_embeddings).flatten()

        # Apply same flip as during fitting
        if self._flip:
            image_1d = -image_1d

        # Find nearest timeline point for each image
        predictions = []
        for point in image_1d:
            distances = np.abs(self.timeline_1d - point)
            nearest_idx = np.argmin(distances)
            predictions.append(self.timeline_years[nearest_idx])

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return np.array(predictions), elapsed_ms

    def evaluate(self, image_embeddings, ground_truth_years):
        """
        Full evaluation on image embeddings.

        Returns:
            dict with predictions, mae, tai, ranking_metrics, timing
        """
        predictions, inference_ms = self.predict(image_embeddings)
        ground_truth_years = np.array(ground_truth_years)

        mae = mean_absolute_error(ground_truth_years, predictions)
        tai_scores = [calculate_TAI(p, g) for p, g in
                      zip(predictions, ground_truth_years)]

        return {
            'predictions': predictions,
            'mae': float(mae),
            'tai': float(np.mean(tai_scores)),
            'timing': {
                'total_inference_ms': inference_ms,
                'avg_per_image_ms': inference_ms / len(image_embeddings),
            },
        }


def main():
    parser = argparse.ArgumentParser(description='UMAP Timeline Evaluation')
    parser.add_argument('--model', type=str, default='clip-vit-b32')
    parser.add_argument('--embeddings_path', type=str, default='encodings')
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--n_trials', type=int, default=100)

    args = parser.parse_args()

    from .embeddings import load_precomputed_embeddings
    data = load_precomputed_embeddings(args.embeddings_path, args.model)

    # Fit timeline
    model = UMAPTimeline()
    model.fit(
        data['timeline_emb'], data['timeline_years'],
        optimize=args.optimize, model_name=args.model,
        n_trials=args.n_trials,
    )

    # Evaluate
    results = model.evaluate(data['image_emb'], data['image_years'])

    print(f"\nResults: MAE={results['mae']:.2f}, TAI={results['tai']:.3f}")
    print_evaluation_summary(
        results['predictions'], data['image_years'],
        f"{args.model} (UMAP Timeline)"
    )


if __name__ == '__main__':
    main()
