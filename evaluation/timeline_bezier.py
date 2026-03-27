"""
Bézier Curve-Based Timeline Representation (Section 3.3.2).

Approximates the chronological sequence of time embeddings with a 1D Bézier
curve C(t), then projects image embeddings onto this curve for temporal inference.

Four variants evaluated in Table 5:
- Bézier(R^N, NN):  Original embedding space, nearest-neighbor prediction
- Bézier(R^N, Int): Original embedding space, interpolation prediction
- Bézier(R^S, NN):  KPCA-reduced subspace (S=13), nearest-neighbor
- Bézier(R^S, Int): KPCA-reduced subspace (S=13), interpolation

Parameters (Section 5.3):
- K = 200 control points (uniformly sampled from sorted time embeddings)
- N_samples = 1000 curve points for fine-grained interpolation
- S = 13 optimal KPCA dimension (Figure 6)
- Kernel: cosine (for KPCA)
"""

import os
import sys
import argparse
import time
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.decomposition import KernelPCA
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import (calculate_TAI, mean_absolute_error,
                           evaluate_chronological_order, compute_ranking_metrics,
                           print_evaluation_summary)


class BezierTimeline:
    """
    Bézier curve-based timeline for temporal inference.

    Fits a smooth 1D curve through chronologically-ordered time embeddings
    using De Casteljau's algorithm, then maps image embeddings to years
    via projection onto the curve.
    """

    def __init__(self, num_control_points=200, num_curve_points=1000):
        """
        Args:
            num_control_points: K in the paper (default 200)
            num_curve_points: N_samples for curve discretization (default 1000)
        """
        self.num_control_points = num_control_points
        self.num_curve_points = num_curve_points

        # Fitted state
        self.control_points = None
        self.curve_points = None
        self.timeline_years = None
        self.year_labels = None  # Interpolated year labels for curve points
        self.reducer = None      # KernelPCA reducer (if using R^S)

    # ── De Casteljau's algorithm (Eq. 2 context) ─────────────────────

    def de_casteljau(self, control_points, t):
        """
        Evaluate Bézier curve at parameter t using De Casteljau's algorithm.

        Args:
            control_points: (K, D) array of control points
            t: Parameter in [0, 1]

        Returns:
            Point on the curve (D,)
        """
        points = control_points.copy()
        n = len(points)
        for r in range(1, n):
            points[:n - r] = (1 - t) * points[:n - r] + t * points[1:n - r + 1]
        return points[0]

    def bezier_curve(self, control_points):
        """Generate discretized Bézier curve with N_samples points."""
        t_values = np.linspace(0, 1, self.num_curve_points)
        return np.array([self.de_casteljau(control_points, t) for t in t_values])

    # ── Projection (Eq. 2) ───────────────────────────────────────────

    def project_onto_curve(self, points, curve):
        """
        Project points onto the Bézier curve via nearest point (Eq. 2).

        T_hat_y = argmin_{t in [0,1]} d(T_y, C(t))

        Args:
            points: (N, D) array of points to project
            curve: (M, D) discretized curve

        Returns:
            projected_points: (N, D) nearest curve points
            projection_indices: (N,) indices into curve array
        """
        projected = []
        indices = []

        for point in points:
            distances = np.linalg.norm(curve - point, axis=1)
            idx = np.argmin(distances)
            projected.append(curve[idx])
            indices.append(idx)

        return np.array(projected), np.array(indices)

    # ── Fitting ──────────────────────────────────────────────────────

    def fit(self, time_embeddings, years, reduce_dim=None):
        """
        Fit Bézier curve to chronologically-sorted time embeddings.

        Args:
            time_embeddings: (T, D) embeddings (assumed sorted by year)
            years: List of T years
            reduce_dim: If set, reduce to S dimensions via KPCA (cosine kernel)

        Returns:
            self
        """
        self.timeline_years = list(years)

        # Dimensionality reduction (R^S variant)
        if reduce_dim is not None and reduce_dim < time_embeddings.shape[1]:
            print(f"Reducing to {reduce_dim}D via Kernel PCA (cosine)...")
            self.reducer = KernelPCA(
                n_components=reduce_dim, kernel='cosine'
            )
            time_embeddings_work = self.reducer.fit_transform(time_embeddings)
        else:
            self.reducer = None
            time_embeddings_work = time_embeddings

        # Select K control points uniformly along the timeline
        K = min(self.num_control_points, len(time_embeddings_work))
        indices = np.linspace(0, len(time_embeddings_work) - 1, K, dtype=int)
        self.control_points = time_embeddings_work[indices]

        # Generate Bézier curve
        print(f"Generating Bézier curve ({K} control points, "
              f"{self.num_curve_points} samples)...")
        self.curve_points = self.bezier_curve(self.control_points)

        # Create interpolated year labels for each curve point
        self.year_labels = np.interp(
            np.linspace(0, len(years) - 1, self.num_curve_points),
            range(len(years)),
            years,
        )

        # Evaluate quality by projecting time embeddings back
        _, proj_indices = self.project_onto_curve(
            time_embeddings_work, self.curve_points
        )
        quality = self._evaluate_quality(proj_indices, years)

        return quality

    def _evaluate_quality(self, projection_indices, years):
        """Evaluate chronological ordering of projected time embeddings."""
        # Order based on curve position
        predicted_order = np.argsort(projection_indices)
        ordered_years = [years[i] for i in predicted_order]

        metrics = compute_ranking_metrics(
            projection_indices, np.arange(len(years))
        )

        print(f"\nBézier timeline quality:")
        print(f"  Spearman ρ:  {metrics['spearman_rho']:.3f}")
        print(f"  Kendall τ:   {metrics['kendall_tau']:.3f}")
        print(f"  δ_MNDL:      {metrics['delta_MNDL']:.3f}")

        return metrics

    # ── Prediction methods ───────────────────────────────────────────

    def predict_nearest_neighbor(self, image_embeddings):
        """
        Predict years using nearest-neighbor matching on the Bézier curve.

        Projects images onto curve, then assigns the year of the closest
        projected time embedding.
        """
        work_emb = self._reduce(image_embeddings)
        _, proj_indices = self.project_onto_curve(work_emb, self.curve_points)

        # Map curve indices to years via the year_labels
        predictions = []
        for idx in proj_indices:
            predictions.append(int(round(self.year_labels[idx])))

        return np.array(predictions)

    def predict_interpolation(self, image_embeddings):
        """
        Predict years using interpolation between adjacent time points (Eq. 3).

        y_pred = (1-w) * y_before + w * y_after
        w = d(I_hat, T_hat_before) / (d(I_hat, T_hat_before) + d(I_hat, T_hat_after))
        """
        work_emb = self._reduce(image_embeddings)
        _, proj_indices = self.project_onto_curve(work_emb, self.curve_points)

        # Use continuous interpolation along the year labels
        predictions = np.interp(
            proj_indices,
            range(len(self.year_labels)),
            self.year_labels,
        ).astype(int)

        return predictions

    def _reduce(self, embeddings):
        """Apply KPCA reduction if fitted."""
        if self.reducer is not None:
            return self.reducer.transform(embeddings)
        return embeddings

    # ── Evaluation ───────────────────────────────────────────────────

    def evaluate(self, image_embeddings, ground_truth_years, method='interpolation'):
        """
        Evaluate timeline predictions.

        Args:
            image_embeddings: (N, D) image embeddings
            ground_truth_years: (N,) true years
            method: 'interpolation' or 'nearest_neighbor'

        Returns:
            dict with predictions, mae, tai, timing
        """
        ground_truth_years = np.array(ground_truth_years)

        t0 = time.perf_counter()
        if method == 'interpolation':
            predictions = self.predict_interpolation(image_embeddings)
        else:
            predictions = self.predict_nearest_neighbor(image_embeddings)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        mae = mean_absolute_error(ground_truth_years, predictions)
        tai_scores = [calculate_TAI(p, g) for p, g in
                      zip(predictions, ground_truth_years)]

        return {
            'predictions': predictions,
            'mae': float(mae),
            'tai': float(np.mean(tai_scores)),
            'timing': {
                'total_inference_ms': elapsed_ms,
                'avg_per_image_ms': elapsed_ms / len(image_embeddings),
            },
        }

    def evaluate_all_variants(self, time_embeddings, time_years,
                              image_embeddings, image_years,
                              reduce_dim=13):
        """
        Evaluate all 4 Bézier variants from Table 5.

        Returns dict mapping variant name to results.
        """
        image_years = np.array(image_years)
        results = {}

        for use_kpca in [False, True]:
            dim_label = f"R^S(S={reduce_dim})" if use_kpca else "R^N"
            rdim = reduce_dim if use_kpca else None

            # Fit
            print(f"\n{'='*60}")
            print(f"Fitting Bézier in {dim_label}...")
            self.fit(time_embeddings, time_years, reduce_dim=rdim)

            for method in ['nearest_neighbor', 'interpolation']:
                method_label = 'NN' if method == 'nearest_neighbor' else 'Int'
                variant = f"Bezier({dim_label},{method_label})"
                print(f"\nEvaluating {variant}...")

                res = self.evaluate(image_embeddings, image_years, method=method)
                res['variant'] = variant
                results[variant] = res

                print(f"  MAE: {res['mae']:.2f}, TAI: {res['tai']:.3f}, "
                      f"Time: {res['timing']['avg_per_image_ms']:.2f}ms/img")

        return results

    # ── Visualization ────────────────────────────────────────────────

    def visualize_3d(self, time_embeddings, years=None, image_embeddings=None,
                     image_years=None, save_path=None):
        """Visualize Bézier curve and embeddings in 3D (via KPCA to 3D)."""
        import matplotlib.pyplot as plt

        kpca_3d = KernelPCA(n_components=3, kernel='cosine')
        emb_3d = kpca_3d.fit_transform(time_embeddings)

        # Fit curve in 3D
        K = min(self.num_control_points, len(emb_3d))
        indices = np.linspace(0, len(emb_3d) - 1, K, dtype=int)
        cp_3d = emb_3d[indices]
        curve_3d = self.bezier_curve(cp_3d)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Time embeddings colored by year
        if years is not None:
            colors = plt.cm.viridis(np.linspace(0, 1, len(emb_3d)))
        else:
            colors = 'steelblue'

        ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2],
                   c=colors, s=20, alpha=0.6, label='Time embeddings')

        # Bézier curve
        ax.plot(curve_3d[:, 0], curve_3d[:, 1], curve_3d[:, 2],
                'r-', linewidth=2, label='Bézier timeline')

        # Image embeddings
        if image_embeddings is not None:
            img_3d = kpca_3d.transform(image_embeddings)
            ax.scatter(img_3d[:, 0], img_3d[:, 1], img_3d[:, 2],
                       c='black', s=10, alpha=0.3, marker='*',
                       label='Image embeddings')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.legend()
        ax.set_title('Bézier Timeline in 3D KPCA Space')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Bézier Timeline Evaluation')
    parser.add_argument('--model', type=str, default='clip-vit-b32')
    parser.add_argument('--embeddings_path', type=str, default='encodings')
    parser.add_argument('--num_control_points', type=int, default=200)
    parser.add_argument('--reduce_dim', type=int, default=13,
                        help='KPCA dimension (0 = no reduction)')
    parser.add_argument('--method', type=str, default='interpolation',
                        choices=['interpolation', 'nearest_neighbor'])
    parser.add_argument('--all_variants', action='store_true',
                        help='Evaluate all 4 variants (Table 5)')
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    from evaluation.embeddings import load_precomputed_embeddings
    data = load_precomputed_embeddings(args.embeddings_path, args.model)

    bezier = BezierTimeline(num_control_points=args.num_control_points)

    if args.all_variants:
        results = bezier.evaluate_all_variants(
            data['timeline_emb'], data['timeline_years'],
            data['image_emb'], data['image_years'],
            reduce_dim=args.reduce_dim,
        )
        print(f"\n{'='*70}")
        print("All Bézier Variants Summary:")
        print(f"{'Variant':<30} {'MAE':>8} {'TAI':>8} {'ms/img':>10}")
        print('-' * 60)
        for name, res in results.items():
            print(f"{name:<30} {res['mae']:>8.2f} {res['tai']:>8.3f} "
                  f"{res['timing']['avg_per_image_ms']:>10.2f}")
    else:
        rdim = args.reduce_dim if args.reduce_dim > 0 else None
        bezier.fit(data['timeline_emb'], data['timeline_years'],
                   reduce_dim=rdim)

        if args.visualize:
            bezier.visualize_3d(
                data['timeline_emb'], data['timeline_years'],
                data['image_emb'], data['image_years'],
                save_path=f'bezier_timeline_{args.model}.png'
            )

        results = bezier.evaluate(
            data['image_emb'], data['image_years'], method=args.method
        )
        print(f"\nResults: MAE={results['mae']:.2f}, TAI={results['tai']:.3f}")
        print_evaluation_summary(
            results['predictions'], data['image_years'],
            f"{args.model} (Bézier-{args.method})"
        )


if __name__ == '__main__':
    main()
