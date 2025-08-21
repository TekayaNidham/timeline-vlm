"""
Main experiment runner for temporal awareness evaluation
Orchestrates time probing and timeline modeling experiments
"""

import os
import sys
import argparse
import yaml
import json
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.time_probing import TimeProbing
from evaluation.timeline_umap import UMAPTimeline
from evaluation.timeline_bezier import BezierTimeline
from data.dataset import TIME10kDataset
from utils.prompts import get_prompt_templates, get_best_prompts_by_model
from utils.metrics import print_evaluation_summary
from models.model_loader import get_available_models


class ExperimentRunner:
    """Main experiment orchestrator"""
    
    def __init__(self, config_path=None):
        """Initialize experiment runner with optional config"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self.get_default_config()
        
        # Create output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_dir / f"results_{timestamp}.json"
        self.results = []
    
    def get_default_config(self):
        """Get default configuration"""
        return {
            'data_path': 'data/TIME10k',
            'output_dir': 'results',
            'device': 'cuda',
            'experiments': {
                'time_probing': {
                    'enabled': True,
                    'models': ['clip-vit-b32', 'eva-clip-l14'],
                    'prompts': ['P7']  # Best performing prompt
                },
                'timeline_umap': {
                    'enabled': True,
                    'models': ['clip-vit-b32', 'eva-clip-l14'],
                    'optimize_params': False,
                    'use_precomputed': True,
                    'embeddings_path': 'encodings'
                },
                'timeline_bezier': {
                    'enabled': True,
                    'models': ['clip-vit-b32', 'eva-clip-l14'],
                    'num_control_points': 200,
                    'reduce_dim': 13,
                    'method': 'interpolation',
                    'use_precomputed': True,
                    'embeddings_path': 'encodings'
                }
            }
        }
    
    def run_time_probing(self):
        """Run time probing experiments"""
        if not self.config['experiments']['time_probing']['enabled']:
            return
        
        print("\n" + "="*80)
        print("Running Time Probing Experiments")
        print("="*80)
        
        # Load dataset
        dataset = TIME10kDataset(self.config['data_path'])
        prompts = get_prompt_templates()
        
        for model_name in self.config['experiments']['time_probing']['models']:
            for prompt_id in self.config['experiments']['time_probing']['prompts']:
                print(f"\nEvaluating {model_name} with prompt {prompt_id}...")
                
                try:
                    # Initialize evaluator
                    evaluator = TimeProbing(model_name, self.config['device'])
                    
                    # Get prompt template
                    prompt_template = prompts[prompt_id]
                    
                    # Run evaluation
                    results = evaluator.evaluate(dataset, prompt_template)
                    
                    # Store results
                    experiment_result = {
                        'experiment': 'time_probing',
                        'model': model_name,
                        'prompt': prompt_id,
                        'mae': results['mae'],
                        'tai': results['tai']
                    }
                    self.results.append(experiment_result)
                    
                    # Print summary
                    print_evaluation_summary(
                        results['predictions'], 
                        results['ground_truths'],
                        f"{model_name} ({prompt_id})"
                    )
                    
                except Exception as e:
                    print(f"Error evaluating {model_name}: {e}")
                    continue
    
    def run_timeline_umap(self):
        """Run UMAP timeline experiments"""
        if not self.config['experiments']['timeline_umap']['enabled']:
            return
        
        print("\n" + "="*80)
        print("Running UMAP Timeline Experiments")
        print("="*80)
        
        umap_config = self.config['experiments']['timeline_umap']
        
        if not umap_config['use_precomputed']:
            print("Warning: Real-time embedding computation not implemented.")
            print("Please use precomputed embeddings.")
            return
        
        import numpy as np
        
        for model_name in umap_config['models']:
            print(f"\nEvaluating UMAP timeline for {model_name}...")
            
            try:
                # Load precomputed embeddings
                if 'eva' in model_name.lower():
                    embeddings_dir = Path(umap_config['embeddings_path']) / 'eva'
                    timeline_emb = np.load(embeddings_dir / 'eva_timeline_embeddings.npy')
                    image_emb = np.load(embeddings_dir / 'eva_image_embeddings.npy')
                    with open(embeddings_dir / 'eva_labels_timeline.txt', 'r') as f:
                        timeline_years = [int(line.strip()) for line in f]
                    with open(embeddings_dir / 'labels.txt', 'r') as f:
                        image_years = [int(line.strip()) for line in f]
                else:
                    embeddings_dir = Path(umap_config['embeddings_path'])
                    timeline_emb = np.load(embeddings_dir / 'timeline_embeddings.npy')
                    image_emb = np.load(embeddings_dir / 'image_embeddings.npy')
                    with open(embeddings_dir / 'timeline_labels.txt', 'r') as f:
                        timeline_years = [int(line.strip()) for line in f]
                    with open(embeddings_dir / 'labels.txt', 'r') as f:
                        image_years = [int(line.strip()) for line in f]
                
                # Create and fit timeline
                timeline_model = UMAPTimeline()
                
                timeline_model.fit(
                    timeline_emb, 
                    timeline_years,
                    optimize=umap_config['optimize_params'],
                    params=None,  # Let fit() determine based on model_name
                    model_name=model_name
                )
                
                # Evaluate
                results = timeline_model.evaluate(image_emb, np.array(image_years))
                
                # Store results
                experiment_result = {
                    'experiment': 'timeline_umap',
                    'model': model_name,
                    'mae': results['mae'],
                    'tai': results['tai']
                }
                self.results.append(experiment_result)
                
                # Print summary
                print_evaluation_summary(
                    results['predictions'],
                    image_years,
                    f"{model_name} (UMAP Timeline)"
                )
                
            except Exception as e:
                print(f"Error with UMAP timeline for {model_name}: {e}")
                continue
    
    def run_timeline_bezier(self):
        """Run Bézier timeline experiments"""
        if not self.config['experiments']['timeline_bezier']['enabled']:
            return
        
        print("\n" + "="*80)
        print("Running Bézier Timeline Experiments")
        print("="*80)
        
        bezier_config = self.config['experiments']['timeline_bezier']
        
        if not bezier_config['use_precomputed']:
            print("Warning: Real-time embedding computation not implemented.")
            print("Please use precomputed embeddings.")
            return
        
        import numpy as np
        
        for model_name in bezier_config['models']:
            print(f"\nEvaluating Bézier timeline for {model_name}...")
            
            try:
                # Load precomputed embeddings
                if 'eva' in model_name.lower():
                    embeddings_dir = Path(bezier_config['embeddings_path']) / 'eva'
                    timeline_emb = np.load(embeddings_dir / 'eva_timeline_embeddings.npy')
                    image_emb = np.load(embeddings_dir / 'eva_image_embeddings.npy')
                    with open(embeddings_dir / 'eva_labels_timeline.txt', 'r') as f:
                        timeline_years = [int(line.strip()) for line in f]
                    with open(embeddings_dir / 'labels.txt', 'r') as f:
                        image_years = [int(line.strip()) for line in f]
                else:
                    embeddings_dir = Path(bezier_config['embeddings_path'])
                    timeline_emb = np.load(embeddings_dir / 'timeline_embeddings.npy')
                    image_emb = np.load(embeddings_dir / 'image_embeddings.npy')
                    with open(embeddings_dir / 'timeline_labels.txt', 'r') as f:
                        timeline_years = [int(line.strip()) for line in f]
                    with open(embeddings_dir / 'labels.txt', 'r') as f:
                        image_years = [int(line.strip()) for line in f]
                
                # Create and fit timeline
                timeline_model = BezierTimeline(
                    num_control_points=bezier_config['num_control_points']
                )
                
                timeline_model.fit(
                    timeline_emb,
                    timeline_years,
                    reduce_dim=bezier_config['reduce_dim']
                )
                
                # Evaluate
                results = timeline_model.evaluate(
                    image_emb,
                    np.array(image_years),
                    method=bezier_config['method']
                )
                
                # Store results
                experiment_result = {
                    'experiment': 'timeline_bezier',
                    'model': model_name,
                    'method': bezier_config['method'],
                    'mae': results['mae'],
                    'tai': results['tai']
                }
                self.results.append(experiment_result)
                
                # Print summary
                print_evaluation_summary(
                    results['predictions'],
                    image_years,
                    f"{model_name} (Bézier Timeline - {bezier_config['method']})"
                )
                
            except Exception as e:
                print(f"Error with Bézier timeline for {model_name}: {e}")
                continue
    
    def save_results(self):
        """Save all results to file"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {self.results_file}")
        
        # Print summary table
        print("\n" + "="*80)
        print("Summary of All Results")
        print("="*80)
        
        print(f"{'Experiment':<20} {'Model':<20} {'MAE':<10} {'TAI':<10}")
        print("-"*60)
        
        for result in self.results:
            exp_name = result['experiment']
            if exp_name == 'time_probing':
                exp_name += f" ({result['prompt']})"
            elif exp_name == 'timeline_bezier':
                exp_name += f" ({result['method'][:3]})"
            
            print(f"{exp_name:<20} {result['model']:<20} "
                  f"{result['mae']:<10.2f} {result['tai']:<10.3f}")
    
    def run_all(self):
        """Run all enabled experiments"""
        print("Starting temporal awareness experiments...")
        print(f"Configuration: {self.config}")
        
        # Run experiments
        self.run_time_probing()
        self.run_timeline_umap()
        self.run_timeline_bezier()
        
        # Save results
        self.save_results()
        
        print("\nAll experiments completed!")


def main():
    parser = argparse.ArgumentParser(description='Run temporal awareness experiments')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data_path', type=str, help='Override data path')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], 
                        help='Override device')
    parser.add_argument('--list_models', action='store_true',
                        help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for model in get_available_models():
            print(f"  - {model}")
        return
    
    # Create runner
    runner = ExperimentRunner(args.config)
    
    # Override config if needed
    if args.data_path:
        runner.config['data_path'] = args.data_path
    if args.output_dir:
        runner.config['output_dir'] = args.output_dir
    if args.device:
        runner.config['device'] = args.device
    
    # Run experiments
    runner.run_all()


if __name__ == '__main__':
    main()