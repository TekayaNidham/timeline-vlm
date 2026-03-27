# configs/

YAML configurations for `scripts/run_experiments.py`.

| Config | Purpose | Requirements |
|---|---|---|
| `full_evaluation.yaml` | All 37 models, all experiments | GPU 32GB+ |
| `lightweight_test.yaml` | CLIP ViT-B/32 only, precomputed embeddings | CPU only |

```bash
python scripts/run_experiments.py --config configs/full_evaluation.yaml
python scripts/run_experiments.py --config configs/lightweight_test.yaml --device cpu
```

See [`docs/reproducing_results.md`](../docs/reproducing_results.md) for full reproduction guide.
