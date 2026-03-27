"""
Main experiment runner for reproducing all results from the paper:

"A Matter of Time: Revealing the Structure of Time in Vision-Language Models"
Tekaya, Waldner, Zeppelzauer (MM '25)

Experiments:
1. Time Probing (Table 1): MAE & TAI for 37 VLMs with P7 prompt
2. Prompt Sensitivity (Table 2): P1-P9 for CLIP and EVA-CLIP
3. Class-wise Analysis (Table 3): Per-category results for EVA-CLIP
4. Chronological Progression (Table 4): KPCA & UMAP 1D ranking metrics
5. Timeline Comparison (Table 5): Time Probing vs UMAP vs 4 Bézier variants
6. Dimension Analysis (Figure 6): MAE per KPCA dimension

Usage:
    # Full evaluation (requires GPU and all models)
    python run_experiments.py --config configs/full_evaluation.yaml

    # Lightweight test (CPU-friendly, subset of models)
    python run_experiments.py --config configs/lightweight_test.yaml

    # Single experiment
    python run_experiments.py --experiment time_probing --models clip-vit-b32

    # List available models
    python run_experiments.py --list_models
"""

import os
import sys
import argparse
import json
import yaml
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model_loader import get_available_models, MODEL_REGISTRY
from utils.prompts import get_prompt_templates
from utils.metrics import (print_evaluation_summary, calculate_mae_per_class,
                           mean_absolute_error, calculate_TAI)


class ExperimentRunner:
    """Orchestrates all paper experiments."""

    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = config.get('device', 'cuda')
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Table 1: Time Probing across all VLMs ────────────────────────

    def run_time_probing(self):
        """Table 1: Evaluate time probing with P7 for all configured models."""
        cfg = self.config.get('time_probing', {})
        if not cfg.get('enabled', True):
            return

        from evaluation.time_probing import TimeProbing
        from evaluation.embeddings import load_precomputed_embeddings

        models = cfg.get('models', ['clip-vit-b32'])
        prompt_id = cfg.get('prompt', 'P7')
        prompts = get_prompt_templates()
        prompt_template = prompts[prompt_id]
        embeddings_path = cfg.get('embeddings_path', 'encodings')
        use_precomputed = cfg.get('use_precomputed', True)

        print("\n" + "=" * 70)
        print(f"Table 1: Time Probing ({prompt_id})")
        print("=" * 70)

        results = []
        for model_name in models:
            print(f"\n  [{model_name}]")
            try:
                if use_precomputed:
                    data = load_precomputed_embeddings(
                        embeddings_path, model_name
                    )
                    evaluator = TimeProbing(model_name, self.device)
                    time_emb = evaluator.encode_time_embeddings(
                        data['timeline_years'], prompt_template
                    )
                    res = evaluator.evaluate_from_embeddings(
                        data['image_emb'], data['image_years'],
                        time_emb, data['timeline_years'],
                    )
                else:
                    from data.dataset import TIME10kDataset
                    dataset = TIME10kDataset(
                        self.config.get('data_path', 'data/TIME10k'),
                        csv_path=self.config.get('csv_path', 'data/time10k.csv'),
                    )
                    evaluator = TimeProbing(model_name, self.device)
                    res = evaluator.evaluate(dataset, prompt_template)

                results.append({
                    'model': model_name,
                    'prompt': prompt_id,
                    'mae': res['mae'],
                    'tai': res['tai'],
                    'timing': res.get('timing', {}),
                })
                print(f"  MAE: {res['mae']:.2f}, TAI: {res['tai']:.3f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        self.results['time_probing'] = results
        self._print_table("Time Probing Results", results,
                          ['model', 'mae', 'tai'])
        return results

    # ── Table 2: Prompt Sensitivity ──────────────────────────────────

    def run_prompt_sensitivity(self):
        """Table 2: Evaluate P1-P9 for CLIP and EVA-CLIP."""
        cfg = self.config.get('prompt_sensitivity', {})
        if not cfg.get('enabled', False):
            return

        from evaluation.time_probing import TimeProbing
        from evaluation.embeddings import load_precomputed_embeddings

        models = cfg.get('models', ['clip-vit-b32', 'eva-clip-l14-336'])
        prompt_ids = cfg.get('prompts', [f'P{i}' for i in range(1, 10)])
        prompts = get_prompt_templates()
        embeddings_path = cfg.get('embeddings_path', 'encodings')

        print("\n" + "=" * 70)
        print("Table 2: Prompt Sensitivity Analysis")
        print("=" * 70)

        results = []
        for model_name in models:
            print(f"\n  [{model_name}]")
            try:
                data = load_precomputed_embeddings(
                    embeddings_path, model_name
                )
                evaluator = TimeProbing(model_name, self.device)

                for pid in prompt_ids:
                    time_emb = evaluator.encode_time_embeddings(
                        data['timeline_years'], prompts[pid]
                    )
                    res = evaluator.evaluate_from_embeddings(
                        data['image_emb'], data['image_years'],
                        time_emb, data['timeline_years'],
                    )
                    results.append({
                        'model': model_name, 'prompt': pid,
                        'mae': res['mae'], 'tai': res['tai'],
                    })
                    print(f"  {pid}: MAE={res['mae']:.2f}, TAI={res['tai']:.3f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        self.results['prompt_sensitivity'] = results
        return results

    # ── Table 4: Chronological Progression in 1D ─────────────────────

    def run_embedding_analysis(self):
        """Table 4: KPCA and UMAP 1D ranking metrics."""
        cfg = self.config.get('embedding_analysis', {})
        if not cfg.get('enabled', False):
            return

        from evaluation.embedding_space import generate_table4
        from evaluation.embeddings import load_precomputed_embeddings

        embeddings_path = cfg.get('embeddings_path', 'encodings')

        print("\n" + "=" * 70)
        print("Table 4: Chronological Progression in 1D")
        print("=" * 70)

        try:
            clip_data = load_precomputed_embeddings(embeddings_path, 'clip-vit-b32')
            eva_data = load_precomputed_embeddings(embeddings_path, 'eva-clip-l14-336')
            results = generate_table4(
                clip_data['timeline_emb'], clip_data['timeline_years'],
                eva_data['timeline_emb'], eva_data['timeline_years'],
            )
            self.results['embedding_analysis'] = {
                str(k): v for k, v in results.items()
            }
        except Exception as e:
            print(f"ERROR: {e}")

    # ── Table 5: Timeline Comparison ─────────────────────────────────

    def run_timeline_comparison(self):
        """Table 5: Time Probing vs UMAP vs 4 Bézier variants."""
        cfg = self.config.get('timeline_comparison', {})
        if not cfg.get('enabled', False):
            return

        from evaluation.time_probing import TimeProbing
        from evaluation.timeline_umap import UMAPTimeline
        from evaluation.timeline_bezier import BezierTimeline
        from evaluation.embeddings import load_precomputed_embeddings

        models = cfg.get('models', ['clip-vit-b32'])
        embeddings_path = cfg.get('embeddings_path', 'encodings')
        reduce_dim = cfg.get('reduce_dim', 13)
        num_control_points = cfg.get('num_control_points', 200)
        prompts = get_prompt_templates()

        print("\n" + "=" * 70)
        print("Table 5: Timeline Comparison")
        print("=" * 70)

        all_results = {}
        for model_name in models:
            print(f"\n  [{model_name}]")
            try:
                data = load_precomputed_embeddings(
                    embeddings_path, model_name
                )
                model_results = {}

                # Time Probing
                print("  Time Probing...")
                evaluator = TimeProbing(model_name, self.device)
                time_emb = evaluator.encode_time_embeddings(
                    data['timeline_years'], prompts['P7']
                )
                tp_res = evaluator.evaluate_from_embeddings(
                    data['image_emb'], data['image_years'],
                    time_emb, data['timeline_years'],
                )
                model_results['Time Probing'] = tp_res

                # UMAP
                print("  UMAP...")
                umap_model = UMAPTimeline()
                umap_model.fit(
                    data['timeline_emb'], data['timeline_years'],
                    model_name=model_name,
                )
                umap_res = umap_model.evaluate(
                    data['image_emb'], data['image_years']
                )
                model_results['UMAP'] = umap_res

                # Bézier (all 4 variants)
                print("  Bézier variants...")
                bezier = BezierTimeline(
                    num_control_points=num_control_points
                )
                bezier_results = bezier.evaluate_all_variants(
                    data['timeline_emb'], data['timeline_years'],
                    data['image_emb'], data['image_years'],
                    reduce_dim=reduce_dim,
                )
                model_results.update(bezier_results)

                all_results[model_name] = model_results

                # Print summary
                print(f"\n  {'Method':<30} {'MAE':>8} {'TAI':>8} {'ms/img':>10}")
                print(f"  {'-'*58}")
                for method, res in model_results.items():
                    ms = res.get('timing', {}).get('avg_per_image_ms', 0)
                    print(f"  {method:<30} {res['mae']:>8.2f} "
                          f"{res['tai']:>8.3f} {ms:>10.2f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        self.results['timeline_comparison'] = {
            model: {method: {'mae': r['mae'], 'tai': r['tai']}
                    for method, r in methods.items()}
            for model, methods in all_results.items()
        }
        return all_results

    # ── Figure 6: Dimension Analysis ─────────────────────────────────

    def run_dimension_analysis(self):
        """Figure 6: MAE per KPCA dimension."""
        cfg = self.config.get('dimension_analysis', {})
        if not cfg.get('enabled', False):
            return

        from evaluation.embedding_space import analyze_dimension_sweep
        from evaluation.embeddings import load_precomputed_embeddings

        embeddings_path = cfg.get('embeddings_path', 'encodings')
        max_dim = cfg.get('max_dim', 50)

        print("\n" + "=" * 70)
        print("Figure 6: Dimension Analysis")
        print("=" * 70)

        for model_name in cfg.get('models', ['clip-vit-b32']):
            try:
                data = load_precomputed_embeddings(
                    embeddings_path, model_name
                )
                results = analyze_dimension_sweep(
                    data['timeline_emb'], data['timeline_years'],
                    data['image_emb'], data['image_years'],
                    max_dim=max_dim,
                )
                self.results[f'dimension_sweep_{model_name}'] = {
                    str(k): v for k, v in results.items()
                }
            except Exception as e:
                print(f"  ERROR for {model_name}: {e}")

    # ── Utilities ────────────────────────────────────────────────────

    def _print_table(self, title, results, columns):
        """Print a formatted results table."""
        if not results:
            return
        headers = [c.upper() for c in columns]
        rows = [[r.get(c, '') for c in columns] for r in results]
        # Format numeric values
        for row in rows:
            for i, val in enumerate(row):
                if isinstance(val, float):
                    row[i] = f"{val:.3f}" if val < 1 else f"{val:.2f}"
        print(f"\n{title}:")
        print(tabulate(rows, headers=headers, tablefmt='grid'))

    def save_results(self):
        """Save all results to JSON."""
        output_path = self.output_dir / f"results_{self.timestamp}.json"
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    def run_all(self):
        """Run all enabled experiments."""
        t0 = time.time()
        print(f"Starting experiments at {self.timestamp}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")

        self.run_time_probing()
        self.run_prompt_sensitivity()
        self.run_embedding_analysis()
        self.run_timeline_comparison()
        self.run_dimension_analysis()

        self.save_results()

        elapsed = time.time() - t0
        print(f"\nAll experiments completed in {elapsed:.0f}s")


def load_config(path):
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Run temporal awareness experiments'
    )
    parser.add_argument('--config', type=str,
                        help='Path to YAML config file')
    parser.add_argument('--experiment', type=str,
                        choices=['time_probing', 'prompt_sensitivity',
                                 'embedding_analysis', 'timeline_comparison',
                                 'dimension_analysis'],
                        help='Run a single experiment')
    parser.add_argument('--models', nargs='+',
                        help='Override model list')
    parser.add_argument('--data_path', type=str,
                        help='Override dataset path')
    parser.add_argument('--embeddings_path', type=str,
                        help='Override embeddings path')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--list_models', action='store_true',
                        help='List all available models')

    args = parser.parse_args()

    if args.list_models:
        print(f"Available models ({len(MODEL_REGISTRY)}):\n")
        for key in get_available_models():
            family, backbone, pretrained = MODEL_REGISTRY[key]
            print(f"  {key:<40} {backbone}")
        return

    # Build config
    if args.config:
        config = load_config(args.config)
    else:
        # Default config for quick testing
        config = {
            'device': args.device,
            'output_dir': args.output_dir,
            'data_path': args.data_path or 'data/TIME10k',
            'csv_path': 'data/time10k.csv',
            'time_probing': {
                'enabled': True,
                'models': args.models or ['clip-vit-b32'],
                'prompt': 'P7',
                'use_precomputed': True,
                'embeddings_path': args.embeddings_path or 'encodings',
            },
            'prompt_sensitivity': {'enabled': False},
            'embedding_analysis': {'enabled': False},
            'timeline_comparison': {'enabled': False},
            'dimension_analysis': {'enabled': False},
        }

    # Apply overrides
    if args.device:
        config['device'] = args.device
    if args.output_dir:
        config['output_dir'] = args.output_dir

    # Enable single experiment if specified
    if args.experiment:
        for exp in ['time_probing', 'prompt_sensitivity', 'embedding_analysis',
                     'timeline_comparison', 'dimension_analysis']:
            if exp not in config:
                config[exp] = {}
            config[exp]['enabled'] = (exp == args.experiment)
        if args.models:
            config[args.experiment]['models'] = args.models
        if args.embeddings_path:
            config[args.experiment]['embeddings_path'] = args.embeddings_path

    runner = ExperimentRunner(config)
    runner.run_all()


if __name__ == '__main__':
    main()
