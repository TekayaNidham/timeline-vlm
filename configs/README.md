# configs/ — Experiment Configurations

YAML configuration files for `run_experiments.py`. Each file defines which experiments to run, which models to use, and all method parameters.

## Files

### `full_evaluation.yaml`

Reproduces **all paper results** — requires GPU with 32GB+ VRAM and all 37 models installed.

- Device: `cuda`
- Runs: Table 1 (37 VLMs), Table 2 (P1–P9), Table 3 (per-class), Table 4 (ranking), Table 5 (timeline comparison), Figure 6 (dimension sweep)
- Parameters: `reduce_dim=13`, `num_control_points=200`

```bash
python run_experiments.py --config configs/full_evaluation.yaml
```

### `lightweight_test.yaml`

**Quick validation** that all pipelines work — runs on CPU with precomputed embeddings, no GPU or model downloads needed.

- Device: `cpu`
- Model: CLIP ViT-B/32 only
- Uses precomputed embeddings from `encodings/`
- Runs all experiment types on subset data
- Takes ~3 minutes on a laptop

```bash
python run_experiments.py --config configs/lightweight_test.yaml
```

## Configuration Structure

```yaml
device: 'cuda'                     # 'cuda' or 'cpu'
output_dir: 'results/my_run'       # Where to save results
data_path: 'data/TIME10k'          # Dataset image directory
csv_path: 'data/time10k.csv'       # Dataset metadata CSV

time_probing:                      # Table 1: Time probing
  enabled: true
  prompt: 'P7'
  use_precomputed: false           # true = use encodings/, false = encode on the fly
  embeddings_path: 'encodings'
  models:
    - clip-vit-b32
    - eva-clip-l14-336

prompt_sensitivity:                # Table 2: Prompt comparison
  enabled: true
  embeddings_path: 'encodings'
  models: [clip-vit-b32, eva-clip-l14-336]
  prompts: ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']

embedding_analysis:                # Table 4: Chronological ordering
  enabled: true
  embeddings_path: 'encodings'

timeline_comparison:               # Table 5: Method comparison
  enabled: true
  embeddings_path: 'encodings'
  reduce_dim: 13
  num_control_points: 200
  models: [clip-vit-b32]

dimension_analysis:                # Figure 6: Dimension sweep
  enabled: true
  embeddings_path: 'encodings'
  max_dim: 50
  models: [clip-vit-b32, eva-clip-l14-336]
```

## Creating Custom Configs

Copy `lightweight_test.yaml` and adjust:
- Set `enabled: false` on experiments you don't need
- Change `models` lists to the models you want to evaluate
- Adjust `reduce_dim` and `num_control_points` to experiment with Bézier parameters
