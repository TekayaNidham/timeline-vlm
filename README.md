# A Matter of Time: Revealing the Structure of Time in Vision-Language Models

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

**Predict the year of any image in 3 lines:**

```python
from timeline_vlm import TimelinePredictor

predictor = TimelinePredictor('clip-vit-b32').fit_from_precomputed('encodings')
print(predictor.predict('photo.jpg'))  # -> 1972
```

**Or from the command line:**

```bash
python predict.py --image photo.jpg
```

No GPU required — precomputed embeddings for CLIP and EVA-CLIP are included.

---

## Installation

```bash
git clone https://github.com/tekayanidham/timeline-vlm.git
cd timeline-vlm
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

For all 37 models (including EVA-CLIP, ImageBind and ViT-Lens):
```bash
bash install_models.sh
```

Verify the installation:
```bash
python scripts/run_experiments.py --config configs/lightweight_test.yaml --device cpu
```

---

## Repository Structure

```
timeline-vlm/
│
│   # ── Use the Framework ─────────────────────────────────────────
├── predict.py                 # Predict year for images (CLI)
├── timeline_vlm.py            # Python API for your pipelines
├── visualize.py               # Visualize timelines and embeddings
│
│   # ── Core Library ──────────────────────────────────────────────
├── evaluation/                # Temporal inference methods
│   ├── time_probing.py        #   Baseline: dot-product similarity (Sec. 3.1)
│   ├── timeline_umap.py       #   UMAP 1D timeline (Sec. 3.3.1)
│   ├── timeline_bezier.py     #   Bezier curve timeline (Sec. 3.3.2)
│   ├── embedding_space.py     #   Embedding analysis (Sec. 3.2)
│   └── embeddings.py          #   Embedding generation & caching
├── models/                    # Unified loader for 37 VLMs
│   └── model_loader.py
├── utils/                     # TAI, MAE, ranking metrics, prompts
│   ├── metrics.py
│   └── prompts.py
├── data/                      # TIME10k dataset loader & downloader
│   ├── dataset.py
│   ├── download.py
│   └── time10k.csv
│
│   # ── Paper Reproduction ────────────────────────────────────────
├── scripts/                   # Benchmark & reproduction scripts
│   ├── reproduce_results.py   #   Per-table: --table 1 2 3 4 5 --figure 6
│   └── run_experiments.py     #   Full YAML-driven experiment pipeline
├── configs/                   # Experiment configurations
│   ├── full_evaluation.yaml   #   All 37 models (GPU)
│   └── lightweight_test.yaml  #   Quick CPU test
├── docs/                      # Extended documentation
│   ├── reproducing_results.md #   Step-by-step reproduction guide
│   ├── methods.md             #   Detailed method descriptions
│   ├── models.md              #   All 37 VLMs documented
│   └── dataset.md             #   TIME10k dataset details
│
│   # ── Data ──────────────────────────────────────────────────────
├── encodings/                 # Precomputed embeddings (CLIP, EVA-CLIP)
└── results/                   # Output directory
```

---

## Which Script Should I Use?

| I want to... | Use this | Example |
|---|---|---|
| **Predict the year of an image** | `predict.py` | `python predict.py --image photo.jpg` |
| **Use this in my Python code** | `timeline_vlm.py` | `from timeline_vlm import TimelinePredictor` |
| **Visualize timelines or embeddings** | `visualize.py` | `python visualize.py timeline` |
| **Reproduce a specific paper table** | `scripts/` | `python scripts/reproduce_results.py --table 5` |
| **Run full benchmark** | `scripts/` | `python scripts/run_experiments.py --config configs/full_evaluation.yaml` |

---

## Predicting Year of First Appearance

### Command Line

```bash
# Default: CLIP ViT-B/32, Bezier R^S method
python predict.py --image photo.jpg

# Choose model and method
python predict.py --image photo.jpg --model eva-clip-l14-336 --method bezier

# Time probing (direct similarity matching)
python predict.py --image photo.jpg --method time_probing --prompt P7

# Batch prediction on a directory
python predict.py --image_dir my_photos/ --output json

# UMAP timeline method
python predict.py --image photo.jpg --method umap

# Save results
python predict.py --image_dir photos/ --output csv --save results.csv
```

### Python API

```python
from timeline_vlm import TimelinePredictor

# Initialize and fit
predictor = TimelinePredictor(
    model='clip-vit-b32',       # Any of the 37 supported models
    method='bezier',            # 'time_probing', 'umap', or 'bezier'
    reduce_dim=13,              # KPCA dimensions (Bezier only)
    bezier_method='interpolation',
)
predictor.fit_from_precomputed('encodings')

# Single prediction
year = predictor.predict('photo.jpg')

# Batch prediction
years = predictor.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# Detailed prediction with confidence
details = predictor.predict_with_details('photo.jpg')

# Evaluate on your own data
results = predictor.evaluate(image_embeddings, ground_truth_years)
print(f"MAE: {results['mae']:.2f}, TAI: {results['tai']:.3f}")
```

---

## Reproducing Paper Results

Reproduction scripts and documentation are separate from the core framework.

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

## Visualizations

```bash
python visualize.py manifold --model clip-vit-b32 --save manifold.png        # 2D/3D embedding manifold
python visualize.py timeline --model clip-vit-b32 --save timeline.png        # 1D KPCA vs UMAP
python visualize.py bezier --model clip-vit-b32 --save bezier.png            # 3D Bezier curve
python visualize.py dimension_sweep --model clip-vit-b32 --save sweep.png    # MAE per dimension
python visualize.py distribution --model clip-vit-b32 --save dist.png        # Year distribution
```

---

## Methods

Three temporal inference approaches, each described in detail in [`docs/methods.md`](docs/methods.md):

| Method | Paper | CLIP MAE | Description |
|---|---|---|---|
| Time Probing | Sec. 3.1 | 9.24 | Dot-product similarity baseline |
| UMAP Timeline | Sec. 3.3.1 | 13.01 | 1D manifold projection |
| **Bezier(R^S, Int)** | **Sec. 3.3.2** | **8.80** | **Bezier curve in KPCA subspace (best)** |

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

- [Paper (ACM Digital Library)](https://dl.acm.org/doi/10.1145/3746027.3758163)
- [arXiv Preprint](https://arxiv.org/pdf/2510.19559)
- [Project Page](https://tekayanidham.github.io/timeline-page/)
- [TIME10k Dataset](https://osf.io/4th79/?view_only=560f540a7bac4d489faf164b16109642)
- [Hugging Face Demo](https://huggingface.co/spaces/Nidhamtek/timeline-vlm)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
