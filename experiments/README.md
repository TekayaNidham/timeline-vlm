# A Matter of Time: Revealing the Structure of Time in Vision-Language Models

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](https://arxiv.org/abs/YOUR_ARXIV_ID)
[![Dataset](https://img.shields.io/badge/dataset-TIME10k-orange.svg)](https://osf.io/4th79/?view_only=560f540a7bac4d489faf164b16109642)

![Teaser](ressources/teaser.png)

This repository contains the official implementation of **"A Matter of Time: Revealing the Structure of Time in Vision-Language Models"**.

## 📄 Abstract

Large-scale vision-language models (VLMs) such as CLIP have gained popularity for their generalizable and expressive multimodal representations. This paper investigates the temporal awareness of VLMs, assessing their ability to position visual content in time. We introduce TIME10k, a benchmark dataset of over 10,000 images with temporal ground truth, and evaluate the time-awareness of 37 VLMs. Our investigation reveals that temporal information is structured along a low-dimensional, non-linear manifold in the VLM embedding space. Based on this insight, we propose methods to derive explicit "timeline" representations from the embedding space.

## 🚀 Key Features

- **TIME10k Dataset**: 10,000+ temporally annotated images across 6 object categories
- **Comprehensive Evaluation**: Time-awareness assessment of 37 state-of-the-art VLMs
- **Novel Timeline Modeling**: UMAP and Bézier curve-based approaches for temporal representation
- **Efficient Inference**: Timeline methods achieve competitive accuracy while being computationally efficient

## 📊 Performance Overview

![Time probing](ressources/performance_scatter.png)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 32GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/time-vlm.git
cd time-vlm
```

2. Create a conda environment:
```bash
conda create -n time-vlm python=3.8
conda activate time-vlm
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install model-specific dependencies:
```bash
bash install_models.sh
```

## 📁 Project Structure

```
time-vlm/
├── configs/              # Configuration files
├── data/                # Dataset utilities
├── models/              # Model implementations
│   ├── clip/           # CLIP variants
│   ├── eva_clip/       # EVA-CLIP models
│   ├── imagebind/      # ImageBind
│   ├── openclip/       # OpenCLIP models
│   └── vit_lens/       # ViT-Lens
├── evaluation/          # Evaluation scripts
│   ├── time_probing.py
│   ├── timeline_umap.py
│   └── timeline_bezier.py
├── encodings/          # Pre-computed embeddings
├── resources/          # Images and figures
├── utils/              # Utility functions
├── requirements.txt
├── install_models.sh
└── README.md
```

## 🏃 Quick Start

### 1. Time Probing Evaluation

Evaluate temporal awareness using prompt-based approach:

```python
python evaluation/time_probing.py \
    --model clip-vit-b32 \
    --data_path /path/to/time10k \
    --output_dir results/
```

### 2. Timeline Modeling

#### UMAP-based Timeline:
```python
python evaluation/timeline_umap.py \
    --model clip-vit-b32 \
    --embeddings_path encodings/ \
    --optimize_params
```

#### Bézier Curve Timeline:
```python
python evaluation/timeline_bezier.py \
    --model eva-clip-l14 \
    --embeddings_path encodings/eva/ \
    --num_control_points 200
```

### 3. Evaluate Multiple Models

```bash
python run_experiments.py --config configs/full_evaluation.yaml
```

## 📈 Results

### Time Probing Performance

| Model | MAE ↓ | TAI ↑ |
|-------|-------|-------|
| ... | ... | ... |

*Full results table to be added*

### Timeline Modeling Results

Our timeline approaches achieve competitive accuracy compared to time probing while being significantly more efficient:

- **UMAP Timeline**: ~50x faster inference
- **Bézier Timeline**: ~100x faster inference

## 🗂️ TIME10k Dataset

The TIME10k dataset contains 10,091 images across 6 categories:
- Cars (4,393)
- Mobile Phones (4,337)
- Ships (841)
- Musical Instruments (436)
- Aircraft (69)
- Weapons & Ammunition (15)

Access the dataset: [TIME10k on OSF](https://osf.io/4th79/?view_only=560f540a7bac4d489faf164b16109642)

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{author2024time,
  title={A Matter of Time: Revealing the Structure of Time in Vision-Language Models},
  author={Anonymous},
  booktitle={Conference},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

todo
## 🙏 Acknowledgments

todo: cultural heritage program? eva clip and the others? 

---

For questions or issues, please open an issue on GitHub or contact [nidham.tekaya@fhstp.ac.at].