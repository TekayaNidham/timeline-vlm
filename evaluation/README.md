# evaluation/ — Temporal Inference Methods

Core implementation of all temporal inference approaches from the paper. Each method takes VLM embeddings and produces year predictions.

## Architecture

```
evaluation/
├── embeddings.py         # Embedding generation & loading (shared by all methods)
├── time_probing.py       # Baseline: dot-product similarity (Section 3.1)
├── timeline_umap.py      # UMAP 1D timeline (Section 3.3.1)
├── timeline_bezier.py    # Bézier curve timeline (Section 3.3.2)
└── embedding_space.py    # Embedding analysis & dimensionality (Section 3.2)
```

**Dependency flow:**

```
embeddings.py ──────────────────────────────────────────────────
      │                                                         │
      ├──→ time_probing.py (loads time + image embeddings)      │
      ├──→ timeline_umap.py (loads time embeddings, fits UMAP)  │
      └──→ timeline_bezier.py (loads time embeddings, fits Bézier)
                                                                │
embedding_space.py ←── timeline_bezier.py, timeline_umap.py ────┘
```

## Files

### `embeddings.py` — Embedding Pipeline

Generates and caches image/text embeddings. This is the data layer that feeds all methods.

```python
from evaluation import load_precomputed_embeddings, generate_time_embeddings

# Load precomputed embeddings (shipped with repo for CLIP and EVA-CLIP)
data = load_precomputed_embeddings('encodings', 'clip-vit-b32')
# data['timeline_emb']   → (325, 512) time embeddings for years 1700-2024
# data['timeline_years'] → [1700, 1701, ..., 2024]
# data['image_emb']      → (9802, 512) image embeddings
# data['image_years']    → [1888, 1972, ...] ground truth years

# Generate time embeddings from scratch
time_emb, years = generate_time_embeddings(
    model, tokenizer, 'clip-vit-b32',
    'was built in the year {year}',
    years=range(1700, 2025), device='cuda',
)
```

### `time_probing.py` — Time Probing Baseline (Section 3.1)

Predicts year via argmax dot-product similarity between image embedding and all year-text embeddings.

**Equation 1:** `y_pred = argmax_y (I^T · T_y)`

```python
from evaluation import TimeProbing

evaluator = TimeProbing('clip-vit-b32', device='cpu')
time_emb = evaluator.encode_time_embeddings(years, 'was built in the year {year}')
result = evaluator.evaluate_from_embeddings(image_emb, image_years, time_emb, years)
# result['mae'] → 9.24, result['tai'] → 0.769
```

### `timeline_umap.py` — UMAP Timeline (Section 3.3.1)

Projects time embeddings onto a 1D manifold using UMAP, then maps image embeddings onto the learned timeline.

**Paper-optimized parameters (Section 5.3):**
- CLIP: `n_neighbors=38, min_dist=0.7446`
- EVA-CLIP: `n_neighbors=21, min_dist=0.1040`

```python
from evaluation import UMAPTimeline

timeline = UMAPTimeline()
quality = timeline.fit(time_emb, years, model_name='clip-vit-b32')
predictions, _ = timeline.predict(image_emb)

# Optimize parameters with Optuna
timeline.optimize_parameters(time_emb, years, n_trials=100)
```

### `timeline_bezier.py` — Bézier Curve Timeline (Section 3.3.2)

Fits a smooth 1D Bézier curve through chronologically-ordered time embeddings using De Casteljau's algorithm, then projects images onto it.

**Four variants (Table 5):**

| Variant | Space | Prediction | Description |
|---|---|---|---|
| Bézier(R^N, NN) | Full | Nearest neighbor | Original embedding space |
| Bézier(R^N, Int) | Full | Interpolation | Original space, Eq. 3 |
| Bézier(R^S, NN) | KPCA | Nearest neighbor | Reduced to S=13 dimensions |
| Bézier(R^S, Int) | KPCA | Interpolation | **Best overall (MAE=8.80)** |

**Parameters (Section 5.3):** K=200 control points, N_samples=1000, S=13 KPCA dimensions, cosine kernel.

```python
from evaluation import BezierTimeline

bezier = BezierTimeline(num_control_points=200)
quality = bezier.fit(time_emb, years, reduce_dim=13)
# quality['spearman_rho'] → 0.999

# Predict
preds = bezier.predict_interpolation(image_emb)   # Eq. 3
preds = bezier.predict_nearest_neighbor(image_emb)

# Evaluate all 4 variants at once
results = bezier.evaluate_all_variants(
    time_emb, years, image_emb, image_years, reduce_dim=13,
)
```

### `embedding_space.py` — Embedding Analysis (Section 3.2)

Analyzes the spatial structure of time embeddings. Produces Table 4 (chronological ordering) and Figure 6 (dimension sweep).

```python
from evaluation.embedding_space import analyze_1d_progression, generate_table4

# Table 4: Ranking quality of 1D projections
metrics = analyze_1d_progression(time_emb, years, method='kpca')
# metrics['spearman_rho'] → 0.96 (CLIP)

# Figure 6: MAE per KPCA dimension
from evaluation.embedding_space import analyze_dimension_sweep
results = analyze_dimension_sweep(time_emb, years, image_emb, image_years, max_dim=50)
```

## Precomputed Embeddings

The `encodings/` directory (repo root) ships precomputed embeddings for **CLIP ViT-B/32** and **EVA-CLIP-L-14-336**, enabling all experiments without GPU or model loading:

```
encodings/
├── sentence_embeddings.npy    # CLIP time embeddings (325, 512)
├── image_embeddings.npy       # CLIP image embeddings (9802, 512)
├── timeline_embeddings.npy    # CLIP timeline embeddings
├── labels.txt                 # CLIP image year labels
├── timeline_labels.txt        # CLIP timeline year labels
├── clipl.npy                  # CLIP auxiliary
└── eva/                       # EVA-CLIP embeddings
    ├── sentence_embeddings.npy
    ├── image_embeddings.npy
    └── ...
```
