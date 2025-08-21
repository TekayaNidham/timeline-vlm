import os
import sys
import argparse
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from tabulate import tabulate

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import calculate_TAI, evaluate_chronological_order


class BezierTimeline:
    """Bézier curve-based timeline modeling"""
    
    def __init__(self, num_control_points=200, num_curve_points=1000):
        self.num_control_points = num_control_points
        self.num_curve_points = num_curve_points
        self.control_points = None
        self.curve_points = None
        self.timeline_years = None
        self.year_labels = None
        
    def de_casteljau(self, control_points, t):
        """De Casteljau's algorithm for Bézier curve generation"""
        points = control_points.copy()
        for r in range(1, len(points)):
            points[:len(points) - r] = (1 - t) * points[:len(points) - r] + t * points[1:len(points) - r + 1]
        return points[0]
    
    def bezier_curve(self, control_points):
        """Generate Bézier curve points"""
        t_values = np.linspace(0, 1, self.num_curve_points)
        curve_points = np.array([self.de_casteljau(control_points, t) for t in t_values])
        return curve_points
    
    def project_onto_curve(self, points, curve):
        """Project points onto Bézier curve"""
        projected_points = []
        projection_indices = []
        
        for point in points:
            distances = np.linalg.norm(curve - point, axis=1)
            nearest_index = np.argmin(distances)
            projected_points.append(curve[nearest_index])
            projection_indices.append(nearest_index)
            
        return np.array(projected_points), np.array(projection_indices)
    
    def fit(self, time_embeddings, years, reduce_dim=None):
        """Fit Bézier curve to time embeddings"""
        self.timeline_years = years
        
        # Optionally reduce dimensionality
        if reduce_dim is not None and reduce_dim < time_embeddings.shape[1]:
            print(f"Reducing dimensionality to {reduce_dim} using Kernel PCA...")
            kpca = KernelPCA(n_components=reduce_dim, kernel='cosine')
            time_embeddings = kpca.fit_transform(time_embeddings)
            self.reducer = kpca
        else:
            self.reducer = None
        
        # Select control points uniformly along the timeline
        indices = np.linspace(0, len(time_embeddings) - 1, self.num_control_points, dtype=int)
        self.control_points = time_embeddings[indices]
        
        # Generate Bézier curve
        self.curve_points = self.bezier_curve(self.control_points)
        
        # Create year labels for the curve
        self.year_labels = np.interp(
            np.linspace(0, len(years) - 1, len(self.curve_points)),
            range(len(years)),
            years
        )
        
        # Project time embeddings onto curve and evaluate quality
        projected_points, indices = self.project_onto_curve(time_embeddings, self.curve_points)
        self._evaluate_timeline_quality(indices, years)
        
        return self
    
    def _evaluate_timeline_quality(self, projection_indices, years):
        """Evaluate chronological ordering quality"""
        # Get predicted order based on projection indices
        predicted_order = np.argsort(projection_indices)
        
        # Calculate correlation metrics
        spearman_corr, _ = spearmanr(range(len(years)), predicted_order)
        kendall_tau, _ = kendalltau(range(len(years)), predicted_order)
        
        # Get ordered years
        ordered_years = [years[i] for i in predicted_order]
        num_swaps, norm_distance = evaluate_chronological_order(ordered_years)
        
        print("\nBézier Timeline Quality Metrics:")
        print(f"Spearman's ρ: {spearman_corr:.3f}")
        print(f"Kendall's τ: {kendall_tau:.3f}")
        print(f"Normalized swap distance: {norm_distance:.3f}")
        
        return {
            'spearman': spearman_corr,
            'kendall': kendall_tau,
            'swap_distance': norm_distance
        }
    
    def predict_nearest_neighbor(self, image_embeddings):
        """Predict years using nearest neighbor on curve"""
        # Reduce dimensionality if needed
        if self.reducer is not None:
            image_embeddings = self.reducer.transform(image_embeddings)
        
        # Project images onto curve
        _, projection_indices = self.project_onto_curve(image_embeddings, self.curve_points)
        
        # Get corresponding years
        predictions = []
        for idx in projection_indices:
            # Find nearest time embedding
            curve_point = self.curve_points[idx]
            if self.reducer is not None:
                time_embeddings_reduced = self.reducer.transform(self.control_points)
            else:
                time_embeddings_reduced = self.control_points
            
            distances = np.linalg.norm(time_embeddings_reduced - curve_point, axis=1)
            nearest_time_idx = np.argmin(distances)
            
            # Map to year
            year_idx = int(nearest_time_idx * (len(self.timeline_years) - 1) / (self.num_control_points - 1))
            predictions.append(self.timeline_years[year_idx])
        
        return np.array(predictions)
    
    def predict_interpolation(self, image_embeddings):
        """Predict years using interpolation between adjacent years"""
        # Reduce dimensionality if needed
        if self.reducer is not None:
            image_embeddings = self.reducer.transform(image_embeddings)
        
        # Project images onto curve
        _, projection_indices = self.project_onto_curve(image_embeddings, self.curve_points)
        
        # Interpolate years
        predictions = np.interp(projection_indices, 
                               range(len(self.year_labels)), 
                               self.year_labels)
        
        return predictions.astype(int)
    
    def evaluate(self, image_embeddings, ground_truth_years, method='interpolation'):
        """Evaluate timeline predictions"""
        if method == 'interpolation':
            predictions = self.predict_interpolation(image_embeddings)
        else:
            predictions = self.predict_nearest_neighbor(image_embeddings)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - ground_truth_years))
        tai_scores = [calculate_TAI(pred, gt) for pred, gt in zip(predictions, ground_truth_years)]
        mean_tai = np.mean(tai_scores)
        
        return {
            'predictions': predictions,
            'mae': mae,
            'tai': mean_tai
        }
    
    def visualize_3d(self, time_embeddings, save_path=None):
        """Visualize the Bézier curve in 3D (requires dimensionality reduction to 3)"""
        if time_embeddings.shape[1] != 3:
            kpca = KernelPCA(n_components=3, kernel='cosine')
            embeddings_3d = kpca.fit_transform(time_embeddings)
        else:
            embeddings_3d = time_embeddings
        
        # Fit curve in 3D
        indices = np.linspace(0, len(embeddings_3d) - 1, self.num_control_points, dtype=int)
        control_points_3d = embeddings_3d[indices]
        curve_3d = self.bezier_curve(control_points_3d)
        
        # Plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot time embeddings
        colors = plt.cm.viridis(np.linspace(0, 1, len(embeddings_3d)))
        ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], 
                  c=colors, s=20, alpha=0.6, label='Time embeddings')
        
        # Plot Bézier curve
        ax.plot(curve_3d[:, 0], curve_3d[:, 1], curve_3d[:, 2], 
               'r-', linewidth=2, label='Bézier timeline')
        
        # Plot control points
        ax.scatter(control_points_3d[:, 0], control_points_3d[:, 1], control_points_3d[:, 2],
                  c='red', s=50, marker='D', label='Control points')
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Bézier Timeline Evaluation')
    parser.add_argument('--model', type=str, default='clip-vit-b32',
                        help='Model name (for loading embeddings)')
    parser.add_argument('--embeddings_path', type=str, required=True,
                        help='Path to pre-computed embeddings')
    parser.add_argument('--num_control_points', type=int, default=200,
                        help='Number of control points for Bézier curve')
    parser.add_argument('--reduce_dim', type=int, default=13,
                        help='Reduce to this dimension (None for no reduction)')
    parser.add_argument('--method', type=str, default='interpolation',
                        choices=['interpolation', 'nearest_neighbor'],
                        help='Prediction method')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the Bézier curve in 3D')
    
    args = parser.parse_args()
    
    # Load embeddings
    if 'eva' in args.model.lower():
        timeline_path = os.path.join(args.embeddings_path, 'eva/eva_timeline_embeddings.npy')
        image_path = os.path.join(args.embeddings_path, 'eva/eva_image_embeddings.npy')
        timeline_labels_path = os.path.join(args.embeddings_path, 'eva/eva_labels_timeline.txt')
        image_labels_path = os.path.join(args.embeddings_path, 'eva/labels.txt')
    else:
        timeline_path = os.path.join(args.embeddings_path, 'timeline_embeddings.npy')
        image_path = os.path.join(args.embeddings_path, 'image_embeddings.npy')
        timeline_labels_path = os.path.join(args.embeddings_path, 'timeline_labels.txt')
        image_labels_path = os.path.join(args.embeddings_path, 'labels.txt')
    
    # Load data
    timeline_embeddings = np.load(timeline_path)
    image_embeddings = np.load(image_path)
    
    with open(timeline_labels_path, 'r') as f:
        timeline_years = [int(line.strip()) for line in f]
    with open(image_labels_path, 'r') as f:
        image_years = [int(line.strip()) for line in f]
    
    # Create timeline model
    timeline_model = BezierTimeline(num_control_points=args.num_control_points)
    
    # Fit timeline
    print(f"Fitting Bézier timeline for {args.model}...")
    timeline_model.fit(timeline_embeddings, timeline_years, reduce_dim=args.reduce_dim)
    
    # Visualize if requested
    if args.visualize:
        timeline_model.visualize_3d(timeline_embeddings, 
                                   save_path=f'bezier_timeline_{args.model}.png')
    
    # Evaluate on images
    print(f"\nEvaluating on image embeddings using {args.method}...")
    results = timeline_model.evaluate(image_embeddings, np.array(image_years), 
                                     method=args.method)
    
    # Print results
    print("\nFinal Results:")
    print(f"MAE: {results['mae']:.2f} years")
    print(f"TAI: {results['tai']:.3f}")
    
    # Print comparison table
    headers = ["Metric", "Method", "Value"]
    table_data = [
        ["Mean Absolute Error (MAE)", args.method, f"{results['mae']:.2f}"],
        ["Time Awareness Index (TAI)", args.method, f"{results['tai']:.3f}"]
    ]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == '__main__':
    main()