# A Matter of Time: Revealing the Structure of Time in Vision-Language Models

[![PyPI](https://img.shields.io/pypi/v/timeline-vlm.svg)](https://pypi.org/project/timeline-vlm/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
<a href="https://arxiv.org/pdf/2510.19559" target="_blank"><img src="https://img.shields.io/badge/arXiv-2510.19559-red.svg" alt="arXiv"></a>
<a href="https://dl.acm.org/doi/10.1145/3746027.3758163" target="_blank"><img src="https://img.shields.io/badge/paper-ACM-blue.svg" alt="Paper"></a>
<a href="https://osf.io/4th79/?view_only=560f540a7bac4d489faf164b16109642" target="_blank"><img src="https://img.shields.io/badge/dataset-TIME10k-orange.svg" alt="Dataset"></a>
<a href="https://huggingface.co/spaces/Nidhamtek/timeline-vlm" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-yellow.svg" alt="Demo"></a>

![Teaser](ressources/teaser.png)

Official implementation of **"A Matter of Time: Revealing the Structure of Time in Vision-Language Models"**, published at ACM Multimedia 2025 (MM '25).

> We investigate the temporal awareness of VLMs, assessing their ability to position visual content in time. We introduce **TIME10k**, a benchmark of over 10,000 images with temporal ground truth, and evaluate **37 VLMs**. We reveal that temporal information is structured along a low-dimensional, non-linear manifold in the VLM embedding space. We propose methods to derive an explicit "timeline" representation using **UMAP** and **Bezier curve** approximation, achieving competitive to superior accuracy while being computationally efficient.

![Performance](ressources/performance_scatter.png)

**Try it now:** A live demo is available on [Hugging Face Spaces](https://huggingface.co/spaces/Nidhamtek/timeline-vlm).

---

## Quick Start

### Install

```bash
pip install timeline-vlm
```

For CLIP models, also install:
```bash
pip install git+https://github.com/openai/CLIP.git
```

### Predict the year of any image in 3 lines:

```python
from timeline_vlm import TimelinePredictor

predictor = TimelinePredictor('clip-vit-b32').fit_from_precomputed('encodings')
print(predictor.predict('photo.jpg'))  # -> 1972
```

### Or from the command line:

```bash
timeline predict photo.jpg
timeline predict photo.jpg --model "CLIP ViT-L/14" --method time_probing
```

### List available models:

```python
from timeline_vlm import list_models
print(list_models())  # 37 supported VLMs

# or from CLI
timeline list-models
```

No GPU required. Precomputed embeddings are automatically downloaded and cached on first use.

---

## Installation

### From PyPI (recommended)

```bash
pip install timeline-vlm
```

### From source (for development or paper reproduction)

```bash
git clone https://github.com/TekayaNidham/timeline-vlm.git
cd timeline-vlm
pip install -e .
```

### Optional dependencies

```bash
# CLIP models (not on PyPI, must be installed separately)
pip install git+https://github.com/openai/CLIP.git

# OpenCLIP models
pip install timeline-vlm[openclip]

# All optional dependencies (excluding CLIP)
pip install timeline-vlm[all]

# For all 37 models (including EVA-CLIP, ImageBind and ViT-Lens)
bash install_models.sh
```

---

## Python API

```python
from timeline_vlm import TimelinePredictor

# Initialize and fit
predictor = TimelinePredictor(
    model='clip-vit-b32',          # Any of the 37 supported models
    method='bezier',               # 'time_probing', 'umap', or 'bezier'
    reduce_dim=None,               # KPCA dimensions (None = original space, 13 = optimal)
    bezier_method='interpolation', # 'interpolation' or 'nearest_neighbor'
)
predictor.fit_from_precomputed('encodings')

# Single prediction
year = predictor.predict('photo.jpg')

# Batch prediction
years = predictor.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# Detailed prediction
details = predictor.predict_with_details('photo.jpg')

# Evaluate on your own data
results = predictor.evaluate(image_embeddings, ground_truth_years)
print(f"MAE: {results['mae']:.2f}, TAI: {results['tai']:.3f}")
```

See [`docs/library.md`](docs/library.md) for the full API reference.

---

## CLI

The package installs two equivalent CLI commands: `timeline` (short) and `timeline-vlm` (full).

```bash
# Predict
timeline predict photo.jpg
timeline predict photos/ --output json --save results.json

# List models (grouped by method support)
timeline list-models
timeline list-models --verbose

# Visualize (1D, 2D, or 3D)
timeline visualize timeline --model clip-vit-b32 --dim 3 --save timeline.png
timeline visualize prediction --image photo.jpg --dim 2 --save pred.png
```

---

## Methods

Three temporal inference approaches, each described in detail in [`docs/methods.md`](docs/methods.md):

| Method | Paper | CLIP MAE | Supported Models |
|---|---|---|---|
| Time Probing | Sec. 3.1 | 9.24 | All 37 models |
| UMAP Timeline | Sec. 3.3.1 | 13.01 | CLIP ViT-B/32, EVA02-CLIP-L/14@336px |
| **Bezier(R^S, Int)** | **Sec. 3.3.2** | **8.80** | **CLIP ViT-B/32, EVA02-CLIP-L/14@336px** |

- **Time probing** works with any model — it only needs the VLM installed.
- **Timeline prediction (Bezier/UMAP)** requires precomputed embeddings. Currently available for CLIP ViT-B/32 and EVA02-CLIP-L/14@336px. More models will be added in upcoming releases.

---

## Reproducing Paper Results

```bash
python scripts/reproduce_results.py --table 5                        # Single table
python scripts/reproduce_results.py --table 4 5                      # Multiple tables
python scripts/reproduce_results.py --figure 6                       # Figure 6
python scripts/reproduce_results.py --all                            # Everything
python scripts/run_experiments.py --config configs/full_evaluation.yaml  # Full benchmark
```

| Flag | What it reproduces |
|---|---|
| `--table 1` | Time probing MAE & TAI for 37 VLMs (P7) |
| `--table 2` | Prompt sensitivity P1-P9 |
| `--table 3` | Class-wise temporal awareness |
| `--table 4` | Chronological ordering quality (KPCA vs UMAP) |
| `--table 5` | Method comparison: Time Probing vs UMAP vs 4 Bezier variants |
| `--figure 6` | MAE per KPCA dimension (optimal S=13) |

See [`docs/reproducing_results.md`](docs/reproducing_results.md) for the full step-by-step guide.

---

## Supported Models (37 VLMs)

| Family | Count | Backend |
|---|---|---|
| CLIP | 9 | `openai/CLIP` |
| EVA-CLIP | 8 | `eva_clip` (BAAI) |
| OpenCLIP | 10 | `open_clip` |
| SigLIP | 3 | `open_clip` |
| Others (CoCa, MobileCLIP, ViTamin, CLIPA, ImageBind, ViT-Lens) | 7 | various |

See [`docs/models.md`](docs/models.md) for the full list with model keys and installation instructions.

---

## Documentation

- [`docs/library.md`](docs/library.md) — Full Python API reference
- [`docs/methods.md`](docs/methods.md) — Detailed method descriptions
- [`docs/models.md`](docs/models.md) — All 37 VLMs documented
- [`docs/dataset.md`](docs/dataset.md) — TIME10k dataset details
- [`docs/reproducing_results.md`](docs/reproducing_results.md) — Step-by-step reproduction guide

---

## Citation

```bibtex
@inproceedings{10.1145/3746027.3758163,
  author = {Tekaya, Nidham and Waldner, Manuela and Zeppelzauer, Matthias},
  title = {A Matter of Time: Revealing the Structure of Time in Vision-Language Models},
  year = {2025},
  isbn = {9798400720352},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3746027.3758163},
  doi = {10.1145/3746027.3758163},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  pages = {12371--12380},
  numpages = {10},
  keywords = {benchmark dataset, multimodal representations, time estimation, time modeling, time reasoning, vision-language models},
  location = {Dublin, Ireland},
  series = {MM '25}
}
```

## Links

- [PyPI Package](https://pypi.org/project/timeline-vlm/)
- [Paper (ACM Digital Library)](https://dl.acm.org/doi/10.1145/3746027.3758163)
- [arXiv Preprint](https://arxiv.org/pdf/2510.19559)
- [Project Page](https://tekayanidham.github.io/timeline-page/)
- [TIME10k Dataset](https://osf.io/4th79/?view_only=560f540a7bac4d489faf164b16109642)
- [Hugging Face Demo](https://huggingface.co/spaces/Nidhamtek/timeline-vlm)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
