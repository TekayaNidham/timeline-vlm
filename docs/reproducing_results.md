# Reproducing Paper Results

Step-by-step guide to reproduce every table and figure from:

> **"A Matter of Time: Revealing the Structure of Time in Vision-Language Models"**
> Tekaya, Waldner, Zeppelzauer — ACM Multimedia 2025 (MM '25)

All commands are run from the **repository root**.

---

## Prerequisites

Precomputed embeddings for **CLIP ViT-B/32** and **EVA-CLIP-L-14-336** are included in `encodings/`. These are sufficient to reproduce all tables on CPU without downloading models or the dataset.

For full reproduction with all 37 models, you need:
- GPU with 32GB+ VRAM
- All models installed (`bash install_models.sh`)
- TIME10k dataset downloaded (`python data/download.py`)

---

## Quick Commands

```bash
python scripts/reproduce_results.py --table 5                        # Single table
python scripts/reproduce_results.py --table 4 5                      # Multiple tables
python scripts/reproduce_results.py --figure 6                       # Figure 6
python scripts/reproduce_results.py --all                            # Everything
python scripts/reproduce_results.py --table 5 --models clip-vit-b32  # Specific model
```

---

## Table 1: Time Probing (37 VLMs)

Evaluates temporal awareness using prompt P7 ("was built in the year {year}") across all 37 VLMs via dot-product similarity (Eq. 1).

```bash
python scripts/reproduce_results.py --table 1
```

**Expected output** (CLIP ViT-B/32): MAE = 9.24, TAI = 0.769

For all 37 models:
```bash
python scripts/run_experiments.py --config configs/full_evaluation.yaml
```

---

## Table 2: Prompt Sensitivity (P1-P9)

Compares 9 prompt formulations for CLIP ViT-B/32 and EVA-CLIP-L-14-336.

```bash
python scripts/reproduce_results.py --table 2
```

**Key finding**: P7 ("was built in the year {year}") consistently performs best across all models.

---

## Table 3: Class-wise Awareness

Per-category temporal awareness for EVA-CLIP.

```bash
python scripts/reproduce_results.py --table 3
```

Note: Full per-class breakdown requires the dataset with class labels. With precomputed embeddings, only aggregate results are shown.

---

## Table 4: Chronological Progression in 1D

Evaluates how well KPCA and UMAP 1D projections preserve chronological order using Spearman's rho, Kendall's tau, and delta_MNDL.

```bash
python scripts/reproduce_results.py --table 4
```

**Expected output:**

| Metric | CLIP KPCA | CLIP UMAP | EVA KPCA | EVA UMAP |
|---|---|---|---|---|
| rho | 0.96 | -0.93 | 0.92 | -0.53 |
| tau | 0.84 | -0.88 | 0.76 | -0.25 |
| delta_MNDL | 0.84 | -0.88 | 0.76 | -0.25 |

---

## Table 5: Timeline Method Comparison

Compares Time Probing, UMAP, and 4 Bezier variants (R^N/R^S x NN/Int).

```bash
python scripts/reproduce_results.py --table 5
```

**Expected output** (CLIP ViT-B/32):

| Method | MAE | TAI | ms/img |
|---|---|---|---|
| Time Probing | 9.24 | 0.769 | 0.03 |
| UMAP | 13.01 | 0.526 | 2.08 |
| Bezier(R^N, NN) | 9.28 | 0.764 | 0.60 |
| Bezier(R^N, Int) | 9.24 | 0.766 | 0.61 |
| Bezier(R^S, NN) | 8.89 | 0.791 | 0.09 |
| **Bezier(R^S, Int)** | **8.80** | **0.795** | **0.09** |

---

## Figure 6: Dimension Analysis

MAE per KPCA dimension, showing optimal S=13.

```bash
python scripts/reproduce_results.py --figure 6 --max_dim 50
```

The generated plot is saved to `results/figure6_dimension_sweep.png`.

---

## Full Benchmark (YAML-driven)

For running all experiments as a single pipeline:

```bash
# Full evaluation — all 37 models, requires GPU
python scripts/run_experiments.py --config configs/full_evaluation.yaml

# Lightweight CPU test (~3 min)
python scripts/run_experiments.py --config configs/lightweight_test.yaml --device cpu

# Single experiment type
python scripts/run_experiments.py --experiment timeline_comparison --models clip-vit-b32
```

See [`configs/`](../configs/) for YAML configuration options.
