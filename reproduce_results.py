"""
Reproduce specific tables and figures from the paper.

One-shot scripts to regenerate each result from the paper:
  Table 1: Time probing MAE & TAI for 37 VLMs
  Table 2: Prompt sensitivity (P1-P9)
  Table 3: Class-wise awareness
  Table 4: Chronological progression in 1D
  Table 5: Timeline method comparison
  Figure 6: Dimension analysis

Usage:
    python reproduce_results.py --table 1           # Single table
    python reproduce_results.py --table 4 5         # Multiple tables
    python reproduce_results.py --figure 6          # Figure 6
    python reproduce_results.py --all               # Everything
    python reproduce_results.py --table 5 --models clip-vit-b32  # Override models
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.metrics import (calculate_TAI, mean_absolute_error,
                           calculate_mae_per_class, print_evaluation_summary)
from utils.prompts import get_prompt_templates


def table1(args):
    """Table 1: Time probing MAE & TAI for VLMs with P7."""
    from evaluation.time_probing import TimeProbing
    from evaluation.embeddings import load_precomputed_embeddings

    prompts = get_prompt_templates()
    models = args.models or ['clip-vit-b32', 'eva-clip-l14-336']

    print("\n" + "=" * 60)
    print("Table 1: Evaluation of time awareness (P7)")
    print("=" * 60)

    rows = []
    for model_name in models:
        try:
            data = load_precomputed_embeddings(args.embeddings_path, model_name)
            evaluator = TimeProbing(model_name, args.device)
            time_emb = evaluator.encode_time_embeddings(
                data['timeline_years'], prompts['P7']
            )
            res = evaluator.evaluate_from_embeddings(
                data['image_emb'], data['image_years'],
                time_emb, data['timeline_years'],
            )
            rows.append([model_name, f"{res['mae']:.2f}", f"{res['tai']:.3f}"])
            print(f"  {model_name}: MAE={res['mae']:.2f}, TAI={res['tai']:.3f}")
        except Exception as e:
            print(f"  {model_name}: ERROR - {e}")

    print("\n" + tabulate(rows, headers=['Model', 'MAE ↓', 'TAI ↑'],
                          tablefmt='grid'))
    return rows


def table2(args):
    """Table 2: Prompt sensitivity (P1-P9) for CLIP and EVA-CLIP."""
    from evaluation.time_probing import TimeProbing
    from evaluation.embeddings import load_precomputed_embeddings

    prompts = get_prompt_templates()
    models = args.models or ['clip-vit-b32', 'eva-clip-l14-336']

    print("\n" + "=" * 60)
    print("Table 2: Prompt Sensitivity")
    print("=" * 60)

    for model_name in models:
        try:
            data = load_precomputed_embeddings(args.embeddings_path, model_name)
            evaluator = TimeProbing(model_name, args.device)

            rows = []
            for pid in sorted(prompts.keys()):
                time_emb = evaluator.encode_time_embeddings(
                    data['timeline_years'], prompts[pid]
                )
                res = evaluator.evaluate_from_embeddings(
                    data['image_emb'], data['image_years'],
                    time_emb, data['timeline_years'],
                )
                marker = ' *' if pid == 'P7' else ''
                rows.append([pid, prompts[pid], f"{res['mae']:.2f}",
                             f"{res['tai']:.3f}{marker}"])

            print(f"\n{model_name}:")
            print(tabulate(rows, headers=['ID', 'Template', 'MAE ↓', 'TAI ↑'],
                           tablefmt='grid'))
        except Exception as e:
            print(f"  {model_name}: ERROR - {e}")


def table3(args):
    """Table 3: Class-wise awareness for EVA-CLIP."""
    from evaluation.time_probing import TimeProbing
    from evaluation.embeddings import load_precomputed_embeddings

    prompts = get_prompt_templates()
    model_name = (args.models or ['eva-clip-l14-336'])[0]

    print("\n" + "=" * 60)
    print(f"Table 3: Class-specific awareness ({model_name})")
    print("=" * 60)
    print("Note: Requires full dataset with class labels for per-class results.")
    print("Using precomputed embeddings (aggregate results only).")

    try:
        data = load_precomputed_embeddings(args.embeddings_path, model_name)
        evaluator = TimeProbing(model_name, args.device)
        time_emb = evaluator.encode_time_embeddings(
            data['timeline_years'], prompts['P7']
        )
        res = evaluator.evaluate_from_embeddings(
            data['image_emb'], data['image_years'],
            time_emb, data['timeline_years'],
        )
        print(f"\n  Aggregate: MAE={res['mae']:.2f}, TAI={res['tai']:.3f}")
        print_evaluation_summary(res['predictions'], res['ground_truths'],
                                 model_name)
    except Exception as e:
        print(f"  ERROR: {e}")


def table4(args):
    """Table 4: Chronological progression in 1D."""
    from evaluation.embedding_space import generate_table4
    from evaluation.embeddings import load_precomputed_embeddings

    print("\n" + "=" * 60)
    print("Table 4: Degree of chronological progression in 1D")
    print("=" * 60)

    clip_data = load_precomputed_embeddings(args.embeddings_path, 'clip-vit-b32')
    eva_data = load_precomputed_embeddings(args.embeddings_path, 'eva-clip-l14-336')
    generate_table4(
        clip_data['timeline_emb'], clip_data['timeline_years'],
        eva_data['timeline_emb'], eva_data['timeline_years'],
    )


def table5(args):
    """Table 5: Time Probing vs UMAP vs Bézier variants."""
    from evaluation.time_probing import TimeProbing
    from evaluation.timeline_umap import UMAPTimeline
    from evaluation.timeline_bezier import BezierTimeline
    from evaluation.embeddings import load_precomputed_embeddings

    prompts = get_prompt_templates()
    models = args.models or ['clip-vit-b32']

    print("\n" + "=" * 60)
    print("Table 5: Timeline method comparison")
    print("=" * 60)

    for model_name in models:
        print(f"\n--- {model_name} ---")
        try:
            data = load_precomputed_embeddings(args.embeddings_path, model_name)
            results = {}

            # Time Probing
            evaluator = TimeProbing(model_name, args.device)
            time_emb = evaluator.encode_time_embeddings(
                data['timeline_years'], prompts['P7']
            )
            tp = evaluator.evaluate_from_embeddings(
                data['image_emb'], data['image_years'],
                time_emb, data['timeline_years'],
            )
            results['Time Probing'] = tp

            # UMAP
            umap_model = UMAPTimeline()
            umap_model.fit(data['timeline_emb'], data['timeline_years'],
                           model_name=model_name)
            umap_res = umap_model.evaluate(data['image_emb'], data['image_years'])
            results['UMAP'] = umap_res

            # Bézier (all 4 variants)
            bezier = BezierTimeline(num_control_points=200)
            bezier_res = bezier.evaluate_all_variants(
                data['timeline_emb'], data['timeline_years'],
                data['image_emb'], data['image_years'],
                reduce_dim=13,
            )
            results.update(bezier_res)

            # Print table
            rows = []
            for method, res in results.items():
                ms = res.get('timing', {}).get('avg_per_image_ms', '-')
                if isinstance(ms, float):
                    ms = f"{ms:.2f}"
                rows.append([method, f"{res['mae']:.2f}", f"{res['tai']:.3f}", ms])

            print("\n" + tabulate(rows,
                                  headers=['Method', 'MAE ↓', 'TAI ↑', 'ms/img'],
                                  tablefmt='grid'))

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


def figure6(args):
    """Figure 6: MAE per KPCA dimension."""
    from evaluation.embedding_space import analyze_dimension_sweep, plot_dimension_sweep
    from evaluation.embeddings import load_precomputed_embeddings

    print("\n" + "=" * 60)
    print("Figure 6: MAE per dimension (KPCA)")
    print("=" * 60)

    models = args.models or ['clip-vit-b32']
    all_results = {}

    for model_name in models:
        try:
            data = load_precomputed_embeddings(args.embeddings_path, model_name)
            print(f"\n{model_name}:")
            results = analyze_dimension_sweep(
                data['timeline_emb'], data['timeline_years'],
                data['image_emb'], data['image_years'],
                max_dim=args.max_dim,
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"  ERROR: {e}")

    if all_results:
        save_path = os.path.join(args.output_dir, 'figure6_dimension_sweep.png')
        first = list(all_results.values())[0]
        second = list(all_results.values())[1] if len(all_results) > 1 else None
        plot_dimension_sweep(first, second, save_path=save_path)


def main():
    parser = argparse.ArgumentParser(
        description='Reproduce specific paper results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--table', nargs='+', type=int,
                        choices=[1, 2, 3, 4, 5],
                        help='Tables to reproduce')
    parser.add_argument('--figure', nargs='+', type=int,
                        choices=[6],
                        help='Figures to reproduce')
    parser.add_argument('--all', action='store_true',
                        help='Reproduce everything')
    parser.add_argument('--models', nargs='+',
                        help='Override model list')
    parser.add_argument('--embeddings_path', type=str, default='encodings')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--max_dim', type=int, default=50,
                        help='Max dimension for Figure 6')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.table and not args.figure and not args.all:
        parser.print_help()
        print("\nExamples:")
        print("  python reproduce_results.py --table 4 5")
        print("  python reproduce_results.py --figure 6")
        print("  python reproduce_results.py --all")
        return

    tables = args.table or []
    figures = args.figure or []

    if args.all:
        tables = [1, 2, 3, 4, 5]
        figures = [6]

    dispatch = {1: table1, 2: table2, 3: table3, 4: table4, 5: table5}
    for t in tables:
        dispatch[t](args)

    if 6 in figures:
        figure6(args)

    print("\nDone!")


if __name__ == '__main__':
    main()
