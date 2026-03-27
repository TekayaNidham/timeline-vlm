# data/ — TIME10k Dataset

This module handles loading and downloading the **TIME10k** benchmark dataset: 10,091 temporally annotated images across 6 categories, spanning 1715–2024.

## Dataset Overview

| Category | Images | Year Range |
|---|---|---|
| Cars | 4,393 | 1888–2024 |
| Mobile Phones | 4,337 | 1984–2024 |
| Ships | 841 | 1744–1999 |
| Musical Instruments | 436 | 1715–2009 |
| Aircraft | 69 | 1893–2017 |
| Weapons & Ammunition | 15 | 1959–2003 |

Images are sourced from Wikimedia Commons and organized into year-based directories:

```
data/TIME10k/
├── 1715/
│   ├── image1.jpg
│   └── ...
├── 1744/
├── ...
└── 2024/
```

## Files

### `dataset.py` — Dataset Loader

Loads TIME10k images with their year labels and (optionally) class labels.

```python
from data import TIME10kDataset

# Load from directory structure
dataset = TIME10kDataset('data/TIME10k')
print(len(dataset))  # 10091
path, year, class_name = dataset[0]

# Load with CSV for class labels
dataset = TIME10kDataset('data/TIME10k', csv_path='data/time10k.csv')

# Filter by class
cars = dataset.filter_by_class('cars')

# Access arrays
years = dataset.years              # np.array of all years
paths = dataset.image_paths        # list of all image paths
labels = dataset.class_labels      # list of class names
```

**Key exports:**
- `TIME10kDataset` — Main dataset class
- `CLASSES` — List of 6 canonical class names
- `CLASS_ALIASES` — Mapping from variant spellings to canonical names
- `normalize_class(raw)` — Normalize a class string to canonical form

### `download.py` — Image Downloader

Downloads images from Wikimedia Commons URLs listed in `time10k.csv`.

```bash
# Download dataset (8 parallel workers by default)
python data/download.py --csv data/time10k.csv --output data/TIME10k

# Faster with more workers
python data/download.py --csv data/time10k.csv --output data/TIME10k --workers 16

# Verify completeness
python data/download.py --csv data/time10k.csv --output data/TIME10k --verify
```

### `time10k.csv` — Dataset Metadata

CSV with columns: `filename`, `year`, `class`, `url`. This file is included in the repo; images must be downloaded separately.

## Alternative: Download from OSF

The full dataset is also available at: https://osf.io/4th79/?view_only=560f540a7bac4d489faf164b16109642
