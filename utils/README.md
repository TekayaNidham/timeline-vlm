# utils/ — Shared Utilities

Metrics and prompt templates used across all evaluation methods.

## Files

### `metrics.py` — Evaluation Metrics

All metrics from the paper, implementing Section 5.4 (TAI) and Section 6.2 (ranking metrics).

**Core metrics:**

```python
from utils import calculate_TAI, mean_absolute_error, compute_ranking_metrics

# TAI — Time Awareness Index (Eq. 4)
# Adaptive tolerance: 20 years for 1700, 5 years for 2024
# Adaptive intolerance: 50 years for 1700, 15 years for 2024
score = calculate_TAI(prediction_year=1975, groundtruth_year=1972)
# Returns float in [0, 1]: 1.0 = perfect, 0.0 = far off

# MAE — Mean Absolute Error in years
mae = mean_absolute_error(gt_years, pred_years)

# Ranking metrics — for evaluating chronological ordering
metrics = compute_ranking_metrics(projected_1d, years)
# metrics['spearman_rho'] → Spearman's rank correlation
# metrics['kendall_tau']  → Kendall's tau
# metrics['delta_MNDL']   → Modified Normalised Damerau-Levenshtein Distance
```

**Additional functions:**
- `calculate_dynamic_thresholds(year)` — Compute T(y) and I(y) for a given year
- `evaluate_chronological_order(projected, years)` — Spearman ρ for 1D projections
- `calculate_mae_per_class(predictions, ground_truths, classes)` — Per-category MAE (Table 3)
- `calculate_mae_per_decade(predictions, ground_truths)` — Error by decade
- `print_evaluation_summary(predictions, ground_truths, model_name)` — Formatted console output

### `prompts.py` — Prompt Templates (Table 2)

Nine prompt formulations (P1–P9) for time probing. P7 is the best performing across all models.

```python
from utils import get_prompt_templates, format_prompt

templates = get_prompt_templates()
# {'P1': '{year}', 'P2': 'year {year}', ..., 'P7': 'was built in the year {year}', ...}

text = format_prompt(templates['P7'], 1972)
# 'was built in the year 1972'
```

Also contains `PROMPT_RESULTS` — a dictionary of paper Table 2 numbers for reference.

## TAI in Detail

The Time Awareness Index uses **adaptive thresholds** that account for the fact that distinguishing years is harder for older objects:

```
TAI(y_pred, y_gt) =
    1.0                               if |y_pred - y_gt| <= T(y_gt)
    1 - (|error| - T) / (I - T)      if T(y_gt) < |error| < I(y_gt)
    0.0                               if |y_pred - y_gt| >= I(y_gt)
```

Where T(y) and I(y) linearly interpolate between:
- **1700:** tolerance = 20 years, intolerance = 50 years
- **2024:** tolerance = 5 years, intolerance = 15 years
