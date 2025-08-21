"""
UMAP-based Timeline Modeling for Temporal Inference
Learns a 1D manifold representation of time using UMAP
"""

import os
import sys
import argparse
import numpy as np
from scipy.stats import spearmanr, kendalltau
import umap
from tabulate import tabulate
import optuna

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import calculate_TAI, calculate_dynamic_thresholds, evaluate_chronological_order


class UMAPTimeline:
    """
    UMAP-based timeline modeling for temporal inference.
    
    Uses optimized parameters from the paper:
    - CLIP models: n_neighbors=38, min_dist=0.7446
    - EVA-CLIP models: n_neighbors=21, min_dist=0.1040
    
    These parameters were found through optimization to maximize
    Spearman correlation between the 1D projection and chronological order.
    """
    
    def __init__(self, n_components=1, metric='cosine', random_state=42):
        self.n_components = n_components
        self.metric = metric
        self.random_state = random_state
        self.umap_model = None
        self.timeline_years = None
        
    def optimize_parameters(self, time_embeddings, years, n_trials=100):
        """Optimize UMAP parameters using Optuna"""
        
        def objective(trial):
            # Suggest hyperparameters
            n_neighbors = trial.suggest_int('n_neighbors', 5, 50)
            min_dist = trial.suggest_float('min_dist', 0.0, 0.99)
            
            # Create UMAP model
            reducer = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=self.metric,
                random_state=self.random_state
            )
            
            # Fit and transform
            embedding_1d = reducer.fit_transform(time_embeddings)
            
            # Calculate Spearman correlation
            predicted_order = np.argsort(embedding_1d.flatten())
            true_order = np.argsort(years)
            
            correlation, _ = spearmanr(predicted_order, true_order)
            
            return -abs(correlation)  # Maximize absolute correlation
        
        # Run optimization
        study = optuna.create_study()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        print(f"Best UMAP parameters: {best_params}")
        
        return best_params
    
    def fit(self, time_embeddings, years, optimize=False, params=None, model_name=None):
        """
        Fit UMAP model to time embeddings.
        
        Args:
            time_embeddings: Time embeddings to fit
            years: Corresponding years
            optimize: Whether to optimize parameters (default: False)
            params: Manual parameters to use (overrides defaults)
            model_name: Model name to determine default parameters
        """
        self.timeline_years = years
        
        if optimize and params is None:
            print("Optimizing UMAP parameters...")
            params = self.optimize_parameters(time_embeddings, years)
        elif params is None:
            # Use paper-optimized parameters based on model
            if model_name and 'eva' in model_name.lower():
                params = {'n_neighbors': 21, 'min_dist': 0.1040}
                print(f"Using EVA-CLIP optimized parameters: {params}")
            else:
                params = {'n_neighbors': 38, 'min_dist': 0.7446}
                print(f"Using CLIP optimized parameters: {params}")
        
        # Create UMAP model with best parameters
        self.umap_model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=params['n_neighbors'],
            min_dist=params['min_dist'],
            metric=self.metric,
            random_state=self.random_state,
            transform_seed=self.random_state
        )
        
        # Fit the model
        self.timeline_embeddings = self.umap_model.fit_transform(time_embeddings)
        
        # Evaluate chronological order
        self._evaluate_timeline_quality()
        
    def _evaluate_timeline_quality(self):
        """Evaluate how well the timeline preserves chronological order"""
        # Get order based on 1D projections
        predicted_order = np.argsort(self.timeline_embeddings.flatten())
        
        # Reorder years based on predicted order
        ordered_years = [self.timeline_years[i] for i in predicted_order]
        
        # Calculate metrics
        spearman_corr, _ = spearmanr(range(len(ordered_years)), predicted_order)
        kendall_tau, _ = kendalltau(range(len(ordered_years)), predicted_order)
        num_swaps, norm_distance = evaluate_chronological_order(ordered_years)
        
        print("\nTimeline Quality Metrics:")
        print(f"Spearman's ρ: {spearman_corr:.3f}")
        print(f"Kendall's τ: {kendall_tau:.3f}")
        print(f"Normalized swap distance: {norm_distance:.3f}")
        
        return {
            'spearman': spearman_corr,
            'kendall': kendall_tau,
            'swap_distance': norm_distance
        }
    
    def predict(self, image_embeddings):
        """Predict years for image embeddings"""
        if self.umap_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Transform image embeddings to 1D
        image_1d = self.umap_model.transform(image_embeddings)
        
        # Find nearest year for each image
        predictions = []
        for img_point in image_1d:
            distances = np.abs(self.timeline_embeddings - img_point)
            nearest_idx = np.argmin(distances)
            predictions.append(self.timeline_years[nearest_idx])
            
        return np.array(predictions)
    
    def evaluate(self, image_embeddings, ground_truth_years):
        """Evaluate timeline predictions"""
        predictions = self.predict(image_embeddings)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - ground_truth_years))
        tai_scores = [calculate_TAI(pred, gt) for pred, gt in zip(predictions, ground_truth_years)]
        mean_tai = np.mean(tai_scores)
        
        return {
            'predictions': predictions,
            'mae': mae,
            'tai': mean_tai
        }


def main():
    parser = argparse.ArgumentParser(description='UMAP Timeline Evaluation')
    parser.add_argument('--model', type=str, default='clip-vit-b32',
                        help='Model name (for loading embeddings)')
    parser.add_argument('--embeddings_path', type=str, required=True,
                        help='Path to pre-computed embeddings')
    parser.add_argument('--optimize_params', action='store_true',
                        help='Optimize UMAP parameters (default: use paper-optimized values)')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of optimization trials')
    
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
    timeline_model = UMAPTimeline()
    
    # Fit timeline
    print(f"Fitting UMAP timeline for {args.model}...")
    timeline_model.fit(
        timeline_embeddings, 
        timeline_years, 
        optimize=args.optimize_params,
        params=None,  # Let fit() determine based on model_name
        model_name=args.model
    )
    
    # Evaluate on images
    print("\nEvaluating on image embeddings...")
    results = timeline_model.evaluate(image_embeddings, np.array(image_years))
    
    # Print results
    print("\nFinal Results:")
    print(f"MAE: {results['mae']:.2f} years")
    print(f"TAI: {results['tai']:.3f}")
    
    # Print comparison table
    headers = ["Metric", "Value"]
    table_data = [
        ["Mean Absolute Error (MAE)", f"{results['mae']:.2f}"],
        ["Time Awareness Index (TAI)", f"{results['tai']:.3f}"]
    ]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == '__main__':
    main()