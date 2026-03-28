# TIME10k Dataset

**TIME10k** is a benchmark dataset of 10,091 temporally annotated images for evaluating the temporal awareness of Vision-Language Models.

## Overview

| Property | Value |
|---|---|
| Total images | 10,091 |
| Year range | 1715–2024 |
| Categories | 6 |
| Source | Wikimedia Commons |
| License | CC-BY / public domain (per-image) |

### Categories

| Category | Images | Year Range |
|---|---|---|
| Cars | 4,393 | 1888–2024 |
| Mobile Phones | 4,337 | 1984–2024 |
| Ships | 841 | 1744–1999 |
| Musical Instruments | 436 | 1715–2009 |
| Aircraft | 69 | 1893–2017 |
| Weapons & Ammunition | 15 | 1959–2003 |

## Download

### Option 1: From Wikimedia Commons URLs

The CSV metadata is included in the repository. Download images:

```bash
python data/download.py --csv data/time10k.csv --output data/TIME10k --workers 16
```

### Option 2: From OSF

Full dataset available at: https://osf.io/4th79/?view_only=560f540a7bac4d489faf164b16109642

## Directory Structure

Images are organized by year:

```
data/TIME10k/
├── 1715/
│   └── image.jpg
├── 1744/
├── ...
└── 2024/
```

## Usage in Code

```python
from timeline_vlm.data import TIME10kDataset

# Load from directory
dataset = TIME10kDataset('data/TIME10k')
path, year, class_name = dataset[0]

# Load with CSV for class labels
dataset = TIME10kDataset('data/TIME10k', csv_path='data/time10k.csv')

# Filter by class
cars = dataset.filter_by_class('cars')

# Access arrays
years = dataset.years
paths = dataset.image_paths
```

## CSV Format

`data/time10k.csv` contains columns: `filename`, `year`, `class`, `url`.

## Precomputed Embeddings

For convenience, precomputed embeddings for CLIP ViT-B/32 and EVA02-CLIP-L/14@336px are available in `encodings/` (auto-downloaded on first use). These enable all experiments without downloading the dataset or models.
