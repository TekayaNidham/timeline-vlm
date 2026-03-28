# utils/

Shared metrics and prompt templates.

| File | Contents |
|---|---|
| `metrics.py` | TAI (Eq. 4), MAE, Spearman, Kendall, delta_MNDL, per-class/decade analysis |
| `prompts.py` | 9 prompt templates P1–P9 (Table 2), P7 = best |

```python
from utils import calculate_TAI, mean_absolute_error, get_prompt_templates
```

See [`docs/methods.md`](../docs/methods.md) for the TAI formula and metric definitions.
