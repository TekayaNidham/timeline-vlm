"""
TIME10k Dataset Loader
Handles loading and iteration over the TIME10k temporal image dataset
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import random


class TIME10kDataset:
    """
    TIME10k dataset containing temporally annotated images.
    
    The dataset structure is:
    data_root/
    ├── 1715/
    │   └── images...
    ├── 1744/
    │   └── images...
    ...
    └── 2024/
        └── images...
    """
    
    def __init__(self, data_root: str, classes: Optional[List[str]] = None):
        """
        Initialize TIME10k dataset.
        
        Args:
            data_root: Root directory of the dataset
            classes: Optional list of classes to filter by
        """
        self.data_root = Path(data_root)
        self.classes = classes or ['Aircraft', 'Cars', 'Mobile_Phones', 
                                  'Music_Instruments', 'Ships', 'Weapons_Ammunition']
        
        # Validate dataset exists
        if not self.data_root.exists():
            raise ValueError(f"Dataset not found at {self.data_root}")
        
        # Load all image paths and years
        self.samples = self._load_samples()
        
        print(f"Loaded TIME10k dataset with {len(self.samples)} images")
        self._print_statistics()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their corresponding years"""
        samples = []
        
        # Iterate through year directories
        for year_dir in sorted(self.data_root.iterdir()):
            if year_dir.is_dir() and year_dir.name.isdigit():
                year = int(year_dir.name)
                
                # Get all images in this year directory
                for img_path in year_dir.iterdir():
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        # Optionally filter by class
                        if self.classes is None or any(cls in img_path.name for cls in self.classes):
                            samples.append((str(img_path), year))
        
        return samples
    
    def _print_statistics(self):
        """Print dataset statistics"""
        year_counts = {}
        for _, year in self.samples:
            year_counts[year] = year_counts.get(year, 0) + 1
        
        min_year = min(year_counts.keys())
        max_year = max(year_counts.keys())
        
        print(f"Year range: {min_year} - {max_year}")
        print(f"Number of unique years: {len(year_counts)}")
        
        # Decade distribution
        decade_counts = {}
        for year, count in year_counts.items():
            decade = (year // 10) * 10
            decade_counts[decade] = decade_counts.get(decade, 0) + count
        
        print("\nImages per decade:")
        for decade in sorted(decade_counts.keys()):
            print(f"  {decade}s: {decade_counts[decade]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample by index"""
        return self.samples[idx]
    
    def __iter__(self):
        """Iterate over all samples"""
        return iter(self.samples)
    
    def get_by_year_range(self, start_year: int, end_year: int) -> List[Tuple[str, int]]:
        """Get samples within a specific year range"""
        return [(path, year) for path, year in self.samples 
                if start_year <= year <= end_year]
    
    def get_by_class(self, class_name: str) -> List[Tuple[str, int]]:
        """Get samples for a specific class"""
        return [(path, year) for path, year in self.samples 
                if class_name.lower() in path.lower()]
    
    def split(self, train_ratio: float = 0.8, seed: int = 42) -> Tuple[List, List]:
        """
        Split dataset into train and test sets.
        
        Args:
            train_ratio: Ratio of training samples
            seed: Random seed for reproducibility
            
        Returns:
            train_samples, test_samples
        """
        random.seed(seed)
        samples_copy = self.samples.copy()
        random.shuffle(samples_copy)
        
        split_idx = int(len(samples_copy) * train_ratio)
        train_samples = samples_copy[:split_idx]
        test_samples = samples_copy[split_idx:]
        
        return train_samples, test_samples
    
    def get_year_distribution(self) -> dict:
        """Get distribution of images per year"""
        year_counts = {}
        for _, year in self.samples:
            year_counts[year] = year_counts.get(year, 0) + 1
        return year_counts
    
    def get_class_distribution(self) -> dict:
        """Get distribution of images per class"""
        class_counts = {cls: 0 for cls in self.classes}
        
        for path, _ in self.samples:
            for cls in self.classes:
                if cls.lower() in path.lower():
                    class_counts[cls] += 1
                    break
        
        return class_counts


def download_time10k(destination: str = "data/TIME10k"):
    """
    Download TIME10k dataset from OSF.
    
    Args:
        destination: Where to save the dataset
    """
    print("TIME10k dataset information:")
    print("Access the dataset at: https://osf.io/4th79/?view_only=560f540a7bac4d489faf164b16109642")
    print("\nThe dataset contains:")
    print("- 10,091 images across 6 categories")
    print("- Temporal annotations from 1715 to 2024")
    print("- Categories: Aircraft, Cars, Mobile Phones, Musical Instruments, Ships, Weapons & Ammunition")
    print("\nPlease download and extract the dataset manually to:", destination)
    
    # Create directory if it doesn't exist
    os.makedirs(destination, exist_ok=True)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="TIME10k Dataset Utilities")
    parser.add_argument('--data_path', type=str, help='Path to TIME10k dataset')
    parser.add_argument('--download', action='store_true', help='Show download instructions')
    
    args = parser.parse_args()
    
    if args.download:
        download_time10k()
    elif args.data_path:
        # Load and show statistics
        dataset = TIME10kDataset(args.data_path)
        
        print(f"\nClass distribution:")
        class_dist = dataset.get_class_distribution()
        for cls, count in class_dist.items():
            print(f"  {cls}: {count}")
    else:
        parser.print_help()