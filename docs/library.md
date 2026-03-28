# Library Reference

Python API reference for `timeline-vlm`. Install with:

```bash
pip install timeline-vlm
```

---

## Quick Start

```python
from timeline_vlm import TimelinePredictor

predictor = TimelinePredictor("CLIP ViT-B/32")
predictor.fit_from_precomputed("encodings")
year = predictor.predict("photo.jpg")
```

---

## TimelinePredictor

The main class for temporal prediction.

```python
from timeline_vlm import TimelinePredictor

predictor = TimelinePredictor(
    model='clip-vit-b32',          # Model name (key or display name)
    method='bezier',               # 'time_probing', 'umap', or 'bezier'
    device=None,                   # 'cuda', 'cpu', or None (auto-detect)
    prompt='P7',                   # Prompt template ID (P1-P9)
    reduce_dim=None,               # KPCA dimensions (None = original space)
    bezier_method='interpolation', # 'interpolation' or 'nearest_neighbor'
    num_control_points=200,        # K parameter for Bezier curve
)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `model` | `'clip-vit-b32'` | Any of 37 supported VLMs. Accepts short keys (`clip-vit-b32`) or display names (`CLIP ViT-B/32`). Case-insensitive. |
| `method` | `'bezier'` | Prediction method. See [Methods](#methods) below. |
| `device` | `None` | `'cuda'` or `'cpu'`. Auto-detects GPU when `None`. |
| `prompt` | `'P7'` | Prompt template for time probing. P7 ("was built in the year {year}") works best. |
| `reduce_dim` | `None` | KPCA dimensions for Bezier. `None` (default) uses the original embedding space (R^N). Set to an integer (e.g. `13`) to reduce. |
| `bezier_method` | `'interpolation'` | How to project onto the Bezier curve. See [Prediction Methods](#prediction-methods). |
| `num_control_points` | `200` | Number of control points K for the Bezier curve. |

---

## Fitting

Before predicting, the predictor needs a fitted timeline. Three options:

### From precomputed embeddings (recommended)

```python
predictor.fit_from_precomputed("encodings")
```

Loads `.npy` files from the given directory. This is the fastest option and does not require the VLM model to be installed.

### From numpy arrays

```python
predictor.fit_from_embeddings(time_embeddings, years)
# time_embeddings: (T, D) numpy array
# years: list of T year values
```

### From scratch (requires model)

```python
predictor.fit_from_dataset(prompt='P7', year_min=1700, year_max=2024)
```

Generates time text embeddings using the VLM. Requires the model library (e.g., `pip install timeline-vlm[clip]`).

All three return `self` for chaining:

```python
year = TimelinePredictor("CLIP ViT-B/32").fit_from_precomputed("encodings").predict("photo.jpg")
```

---

## Prediction

### Single image

```python
year = predictor.predict("photo.jpg")          # Returns int
year = predictor.predict(pil_image)             # PIL Image
year = predictor.predict(numpy_array)           # numpy array (H, W, 3)
```

### Batch prediction

```python
years = predictor.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
# Returns numpy array of predicted years
```

### Detailed prediction

```python
details = predictor.predict_with_details("photo.jpg")
# {
#     'predicted_year': 1972,
#     'model': 'clip-vit-b32',
#     'method': 'bezier',
#     'inference_ms': 45.2,
#     'top_predictions': [...]  # only for time_probing
# }
```

---

## Methods

Three temporal inference approaches, corresponding to the paper sections:

### Time Probing (`method='time_probing'`)

Direct dot-product similarity between image and year-text embeddings (Eq. 1):

```
y_pred = argmax_y (I^T . T_y)
```

```python
predictor = TimelinePredictor(model='clip-vit-b32', method='time_probing', prompt='P7')
```

### UMAP Timeline (`method='umap'`)

1D manifold projection using UMAP with model-specific optimized parameters (Section 3.3.1):

```python
predictor = TimelinePredictor(model='clip-vit-b32', method='umap')
```

### Bezier Timeline (`method='bezier'`)

Bezier curve fitted through the embedding space using De Casteljau's algorithm with K=200 control points (Section 3.3.2):

```python
predictor = TimelinePredictor(
    model='clip-vit-b32',
    method='bezier',
    reduce_dim=None,               # None = original space; set to 13 for optimal reduced space
    bezier_method='interpolation', # or 'nearest_neighbor'
    num_control_points=200,
)
```

---

## Prediction Methods (Bezier)

The Bezier timeline supports two ways to project a query embedding onto the curve:

### Interpolation (`bezier_method='interpolation'`)

Weighted interpolation along the curve parameter (Eq. 3). For each query point, computes a weighted combination of curve parameters based on distances to control points. This is the default and generally gives the best results.

```python
predictor = TimelinePredictor(method='bezier', bezier_method='interpolation')
```

### Nearest Neighbor (`bezier_method='nearest_neighbor'`)

Euclidean distance to curve. Finds the closest point on the Bezier curve to the query embedding and reads off the corresponding year.

```python
predictor = TimelinePredictor(method='bezier', bezier_method='nearest_neighbor')
```

### Dimension Control (`reduce_dim`)

The Bezier method can operate in the original embedding space (R^N, e.g., R^512 for CLIP ViT-B/32) or in a reduced KPCA space (R^S):

| `reduce_dim` | Space | Description |
|---|---|---|
| `None` (default) | R^N | Original embedding space. No dimensionality reduction. |
| `13` | R^S (S=13) | Optimal dimensionality per Figure 6. Removes noise, best MAE. |
| Any integer S > 0 | R^S | Custom KPCA dimensionality. |

This gives four Bezier variants as reported in the paper (Table 2):

```python
# R^N + Interpolation (default — original embedding space)
TimelinePredictor(method='bezier', bezier_method='interpolation')

# R^N + Nearest Neighbor
TimelinePredictor(method='bezier', bezier_method='nearest_neighbor')

# R^S + Interpolation (reduced space, best MAE)
TimelinePredictor(method='bezier', reduce_dim=13, bezier_method='interpolation')

# R^S + Nearest Neighbor
TimelinePredictor(method='bezier', reduce_dim=13, bezier_method='nearest_neighbor')
```

---

## Visualization

Visualize timelines and predictions in 1D, 2D, or 3D using KPCA projections.

```python
from timeline_vlm.visualization import plot_prediction, plot_timeline
```

### Plot a prediction

```python
result = plot_prediction(predictor, "photo.jpg", dim=3, save_path="pred.png")
# result = {'predicted_year': 1972, 'figure': <Figure>}
```

### Plot the timeline

```python
fig = plot_timeline(predictor, dim=3, save_path="timeline.png")
```

### Dimension parameter

| `dim` | Description |
|---|---|
| `1` | 1D: KPCA projection vs year (scatter plot) |
| `2` | 2D: PC1 vs PC2 colored by year |
| `3` (default) | 3D: Interactive 3D scatter with PC1/PC2/PC3 |

```python
plot_timeline(predictor, dim=1, save_path="timeline_1d.png")
plot_timeline(predictor, dim=2, save_path="timeline_2d.png")
plot_timeline(predictor, dim=3, save_path="timeline_3d.png")
```

---

## Evaluation

Evaluate predictions against ground truth:

```python
results = predictor.evaluate(image_embeddings, ground_truth_years)
# {
#     'mae': 8.80,    # Mean Absolute Error (years)
#     'tai': 0.72,    # Time Awareness Index (Eq. 4)
#     'predictions': array([...])
# }
```

---

## Model Selection

37 VLMs across 5 families. List them from Python or the CLI:

```python
from timeline_vlm import list_models

# Display names
models = list_models()
# ['CLIP RN50', 'CLIP RN101', 'CLIP ViT-B/16', ...]

# With details
models = list_models(verbose=True)
# {'CLIP ViT-B/32': {'key': 'clip-vit-b32', 'family': 'clip', ...}, ...}

# Also available as a static method on the predictor
models = TimelinePredictor.list_models()
models = TimelinePredictor.list_models(verbose=True)
```

```bash
timeline-vlm list-models
timeline-vlm list-models --verbose
```

Models can be referenced by key or display name:

```python
# These are equivalent:
TimelinePredictor(model='clip-vit-b32')
TimelinePredictor(model='CLIP ViT-B/32')
```

### Model families

| Family | Count | Example |
|---|---|---|
| CLIP | 9 | `CLIP ViT-B/32`, `CLIP ViT-L/14@336px` |
| EVA-CLIP | 8 | `EVA-CLIP ViT-B/16` |
| OpenCLIP | 10 | `OpenCLIP ViT-bigG/14` |
| SigLIP | 3 | `SigLIP ViT-B/16@384px` |
| Other | 7 | `BLIP-2 ViT-L`, `DINOv2 ViT-L/14` |

> More models will be added in upcoming releases.

---

## Saving and Loading Embeddings

Save fitted timeline embeddings for reuse:

```python
predictor.save_embeddings("my_encodings/")
```

Load them later:

```python
predictor.fit_from_precomputed("my_encodings/")
```

---

## CLI

Installed as `timeline-vlm` when the package is pip-installed.

### Predict

```bash
timeline-vlm predict photo.jpg
timeline-vlm predict photo.jpg --model "CLIP ViT-L/14" --method time_probing
timeline-vlm predict photos/ --output json --save results.json
```

### List models

```bash
timeline-vlm list-models
timeline-vlm list-models --verbose
```

### Visualize

```bash
timeline-vlm visualize timeline --model clip-vit-b32 --dim 3 --save timeline.png
timeline-vlm visualize prediction --image photo.jpg --dim 2 --save pred.png
timeline-vlm visualize timeline --dim 1
```

| Flag | Default | Description |
|---|---|---|
| `--dim` | `3` | Projection dimensionality: 1, 2, or 3 |
| `--model` | `clip-vit-b32` | Model to use |
| `--device` | `cpu` | Device (`cpu` or `cuda`) |
| `--embeddings_path` | `encodings` | Path to precomputed embeddings |
| `--image` | — | Image path (required for `prediction` type) |
| `--save` | — | Save figure to file |

---

## Properties

```python
predictor.is_fitted         # bool — whether fit has been called
predictor.timeline_years    # list of years in the fitted timeline
predictor.timeline_quality  # dict with fit quality metrics (Bezier/UMAP only)
predictor.model_name        # resolved model key
```

---

## Convenience Function

One-liner for quick predictions:

```python
from timeline_vlm import predict_year

year = predict_year("photo.jpg", model="clip-vit-b32", method="bezier",
                    embeddings_path="encodings")
```
