# data/

TIME10k dataset loading and downloading.

| File | Purpose |
|---|---|
| `dataset.py` | `TIME10kDataset` class — loads images with year labels and class names |
| `download.py` | Downloads images from Wikimedia Commons URLs in `time10k.csv` |
| `time10k.csv` | Dataset metadata (10,091 entries) |

```python
from data import TIME10kDataset, CLASSES

dataset = TIME10kDataset('data/TIME10k', csv_path='data/time10k.csv')
path, year, class_name = dataset[0]
```

```bash
python data/download.py --csv data/time10k.csv --output data/TIME10k --workers 16
```

See [`docs/dataset.md`](../docs/dataset.md) for full dataset documentation.
