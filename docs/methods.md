# Methods

Overview of the three temporal inference approaches implemented in this framework. Each method takes VLM embeddings and produces year-of-first-appearance predictions.

---

## Time Probing (Section 3.1) — Baseline

Direct dot-product similarity between an image embedding and year-text embeddings.

**Equation 1:** `y_pred = argmax_y (I^T . T_y)`

Each year y in [1700, 2024] is encoded as text using a prompt template. The predicted year is the one whose text embedding has the highest cosine similarity to the image embedding.

- **Prompt**: P7 ("was built in the year {year}") performs best across all 37 VLMs
- **Complexity**: O(T) per image, where T = number of years
- **Implementation**: [`evaluation/time_probing.py`](../evaluation/time_probing.py)

```python
from evaluation import TimeProbing

evaluator = TimeProbing('clip-vit-b32', device='cpu')
time_emb = evaluator.encode_time_embeddings(years, 'was built in the year {year}')
result = evaluator.evaluate_from_embeddings(image_emb, image_years, time_emb, years)
```

---

## UMAP Timeline (Section 3.3.1)

Projects time embeddings onto a 1D manifold using UMAP, then maps image embeddings onto the learned timeline.

**Paper-optimized parameters (Section 5.3):**
- CLIP ViT-B/32: `n_neighbors=38, min_dist=0.7446`
- EVA-CLIP-L-14-336: `n_neighbors=21, min_dist=0.1040`
- Metric: cosine
- Optimization: TPE via Optuna, maximizing Spearman's rank correlation

**Implementation**: [`evaluation/timeline_umap.py`](../evaluation/timeline_umap.py)

```python
from evaluation import UMAPTimeline

timeline = UMAPTimeline()
quality = timeline.fit(time_emb, years, model_name='clip-vit-b32')
predictions, _ = timeline.predict(image_emb)
```

---

## Bezier Curve Timeline (Section 3.3.2)

Fits a smooth 1D Bezier curve C(t) through chronologically-ordered time embeddings using De Casteljau's algorithm, then projects images onto the curve for year prediction.

**Four variants (Table 5):**

| Variant | Space | Prediction | Description |
|---|---|---|---|
| Bezier(R^N, NN) | Full (512-d) | Nearest neighbor | Project to nearest curve point |
| Bezier(R^N, Int) | Full (512-d) | Interpolation (Eq. 3) | Weighted interpolation of neighbors |
| Bezier(R^S, NN) | KPCA (S-d) | Nearest neighbor | Reduced space, nearest point |
| **Bezier(R^S, Int)** | **KPCA (S-d)** | **Interpolation (Eq. 3)** | **Best overall** |

**Parameters (Section 5.3):**
- K = 200 control points (uniformly sampled from sorted time embeddings)
- N_samples = 1000 curve points for fine-grained interpolation
- S = 13 optimal KPCA dimension (Figure 6)
- Kernel: cosine (for KPCA)

**Implementation**: [`evaluation/timeline_bezier.py`](../evaluation/timeline_bezier.py)

```python
from evaluation import BezierTimeline

bezier = BezierTimeline(num_control_points=200)
quality = bezier.fit(time_emb, years, reduce_dim=13)
preds = bezier.predict_interpolation(image_emb)
```

---

## Metrics

### MAE — Mean Absolute Error

Mean absolute difference in years between prediction and ground truth. Lower is better.

### TAI — Time Awareness Index (Eq. 4)

Adaptive tolerance metric that accounts for the difficulty of dating older objects:

```
TAI(y_pred, y_gt) =
    1.0                           if |error| <= T(y_gt)
    1 - (|error| - T) / (I - T)  if T(y_gt) < |error| < I(y_gt)
    0.0                           if |error| >= I(y_gt)
```

Where T(y) and I(y) linearly interpolate:
- **1700**: tolerance = 20 years, intolerance = 50 years
- **2024**: tolerance = 5 years, intolerance = 15 years

### Ranking Metrics

For evaluating chronological ordering quality of 1D projections:
- **Spearman's rho**: Rank correlation between projected positions and years
- **Kendall's tau**: Concordance of pairwise orderings
- **delta_MNDL**: Modified Normalised Damerau-Levenshtein Distance = 1 - 2S/M

**Implementation**: [`utils/metrics.py`](../utils/metrics.py)
