# evaluation/

Core temporal inference methods from the paper.

| File | Method | Paper Section |
|---|---|---|
| `time_probing.py` | Time Probing (dot-product similarity baseline) | Section 3.1 |
| `timeline_umap.py` | UMAP 1D timeline projection | Section 3.3.1 |
| `timeline_bezier.py` | Bezier curve timeline (4 variants) | Section 3.3.2 |
| `embedding_space.py` | Embedding analysis & dimensionality | Section 3.2 |
| `embeddings.py` | Embedding generation, caching & loading | — |

```python
from evaluation import TimeProbing, UMAPTimeline, BezierTimeline
from evaluation import load_precomputed_embeddings
```

See [`docs/methods.md`](../docs/methods.md) for detailed explanations of each method.
