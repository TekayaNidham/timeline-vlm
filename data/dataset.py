"""
TIME10k Dataset Loader.

Supports two modes:
1. Directory-based: Load from year-based directory structure (data/TIME10k/{year}/...)
2. CSV-based: Load metadata from time10k.csv with image paths resolved from directory

The dataset contains 10,091 temporally annotated images across 6 categories:
- Aircraft (69), Cars (4,393), Mobile Phones (4,337),
- Musical Instruments (436), Ships (841), Weapons & Ammunition (15)
- Year range: 1715-2024
"""

import os
import csv
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import numpy as np


# Canonical class names
CLASSES = [
    'aircraft', 'cars', 'mobile_phones',
    'music_instruments', 'ships', 'weapons_ammunition',
]

# For matching filenames/CSV entries to canonical class names
CLASS_ALIASES = {
    'aircraft': ['aircraft'],
    'cars': ['cars', 'car'],
    'mobile_phones': ['mobile_phones', 'mobile phones', 'mobilephones', 'phone'],
    'music_instruments': ['music_instruments', 'musical_instruments',
                          'music instruments', 'musical instruments',
                          'instruments'],
    'ships': ['ships', 'ship'],
    'weapons_ammunition': ['weapons_ammunition', 'weapons & ammunition',
                           'weapons and ammunition', 'weapons', 'ammunition'],
}


def normalize_class(raw_class):
    """Normalize a class string to canonical form."""
    raw_lower = raw_class.lower().strip().strip('"')
    for canonical, aliases in CLASS_ALIASES.items():
        for alias in aliases:
            if alias in raw_lower:
                return canonical
    return raw_lower


class TIME10kDataset:
    """
    TIME10k dataset for temporal awareness evaluation.

    Each sample is a tuple of (image_path, year, class_name).
    """

    def __init__(self, data_root: str, csv_path: Optional[str] = None,
                 classes: Optional[List[str]] = None):
        """
        Initialize TIME10k dataset.

        Args:
            data_root: Root directory containing year-based subdirectories
            csv_path: Optional path to time10k.csv for metadata
            classes: Optional list of classes to filter by
        """
        self.data_root = Path(data_root)
        self.csv_path = csv_path
        self.filter_classes = [c.lower() for c in classes] if classes else None

        if csv_path and os.path.exists(csv_path):
            self.samples = self._load_from_csv()
        elif self.data_root.exists():
            self.samples = self._load_from_directory()
        else:
            raise ValueError(
                f"Dataset not found at {self.data_root}. "
                f"Run: python data/download.py --csv data/time10k.csv "
                f"--output {self.data_root}"
            )

        if not self.samples:
            raise ValueError(f"No images found in {self.data_root}")

        print(f"Loaded TIME10k: {len(self.samples)} images")

    def _load_from_csv(self) -> List[Tuple[str, int, str]]:
        """Load samples using CSV metadata, matching to files on disk."""
        from .download import sanitize_filename, load_csv

        entries = load_csv(self.csv_path)
        samples = []

        for url, year, cls, obj_name in entries:
            cls_norm = normalize_class(cls)
            if self.filter_classes and cls_norm not in self.filter_classes:
                continue

            filename = sanitize_filename(url)
            img_path = self.data_root / str(year) / filename
            if img_path.exists():
                samples.append((str(img_path), year, cls_norm))

        return samples

    def _load_from_directory(self) -> List[Tuple[str, int, str]]:
        """Load samples by scanning year directories."""
        samples = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        for year_dir in sorted(self.data_root.iterdir()):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            year = int(year_dir.name)

            for img_path in year_dir.iterdir():
                if img_path.suffix.lower() not in valid_extensions:
                    continue

                # Infer class from filename
                cls = self._infer_class(img_path.name)
                if self.filter_classes and cls not in self.filter_classes:
                    continue

                samples.append((str(img_path), year, cls))

        return samples

    def _infer_class(self, filename: str) -> str:
        """Infer class from filename using heuristics."""
        fname_lower = filename.lower()
        for canonical, aliases in CLASS_ALIASES.items():
            for alias in aliases:
                if alias in fname_lower:
                    return canonical
        return 'unknown'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    @property
    def image_paths(self) -> List[str]:
        return [s[0] for s in self.samples]

    @property
    def years(self) -> np.ndarray:
        return np.array([s[1] for s in self.samples])

    @property
    def class_labels(self) -> List[str]:
        return [s[2] for s in self.samples]

    def get_by_year_range(self, start_year: int, end_year: int):
        """Get samples within a year range."""
        return [(p, y, c) for p, y, c in self.samples
                if start_year <= y <= end_year]

    def get_by_class(self, class_name: str):
        """Get samples for a specific class."""
        class_name = normalize_class(class_name)
        return [(p, y, c) for p, y, c in self.samples if c == class_name]

    def split(self, train_ratio: float = 0.8, seed: int = 42):
        """Split into train/test sets with reproducible seeding."""
        rng = random.Random(seed)
        samples_copy = self.samples.copy()
        rng.shuffle(samples_copy)
        split_idx = int(len(samples_copy) * train_ratio)
        return samples_copy[:split_idx], samples_copy[split_idx:]

    def get_statistics(self) -> Dict:
        """Get comprehensive dataset statistics."""
        year_counts = defaultdict(int)
        class_counts = defaultdict(int)
        class_year_ranges = defaultdict(lambda: [9999, 0])

        for _, year, cls in self.samples:
            year_counts[year] += 1
            class_counts[cls] += 1
            r = class_year_ranges[cls]
            r[0] = min(r[0], year)
            r[1] = max(r[1], year)

        return {
            'total_images': len(self.samples),
            'year_range': (min(year_counts), max(year_counts)),
            'num_unique_years': len(year_counts),
            'class_counts': dict(class_counts),
            'class_year_ranges': {k: tuple(v) for k, v
                                  in class_year_ranges.items()},
            'year_counts': dict(year_counts),
        }

    def print_statistics(self):
        """Print formatted dataset statistics."""
        stats = self.get_statistics()
        print(f"\nTIME10k Dataset Statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Year range: {stats['year_range'][0]}-{stats['year_range'][1]}")
        print(f"  Unique years: {stats['num_unique_years']}")
        print(f"\n  Class distribution:")
        for cls, count in sorted(stats['class_counts'].items(),
                                 key=lambda x: -x[1]):
            yr = stats['class_year_ranges'].get(cls, ('?', '?'))
            print(f"    {cls:<25} {count:>5}  [{yr[0]}-{yr[1]}]")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TIME10k Dataset Utilities')
    parser.add_argument('--data_path', type=str, default='data/TIME10k',
                        help='Path to TIME10k dataset')
    parser.add_argument('--csv', type=str, default='data/time10k.csv',
                        help='Path to time10k.csv')
    parser.add_argument('--stats', action='store_true',
                        help='Print dataset statistics')

    args = parser.parse_args()

    if args.stats or args.data_path:
        csv_path = args.csv if os.path.exists(args.csv) else None
        dataset = TIME10kDataset(args.data_path, csv_path=csv_path)
        dataset.print_statistics()
