"""
Evaluation metrics for temporal awareness assessment.

Includes:
- TAI (Time Awareness Index) with adaptive tolerance (Eq. 4 in paper)
- Chronological ordering metrics: Spearman's rho, Kendall's tau, delta_MNDL
- MAE and per-decade error analysis
"""

import numpy as np
from scipy.stats import spearmanr, kendalltau


def calculate_dynamic_thresholds(year, Y_min=1700, Y_max=2024,
                                  T_old=20, T_recent=5, I_old=50, I_recent=15):
    """
    Calculate dynamic tolerance and intolerance thresholds based on year.

    From Section 5.4 of the paper:
    - T(y): tolerance function, linearly interpolated from T_old to T_recent
    - I(y): intolerance function, linearly interpolated from I_old to I_recent

    Paper parameters: T_Ymin=20, I_Ymin=50, T_Ymax=5, I_Ymax=15

    Args:
        year: The ground truth year
        Y_min: Earliest year in range (1700)
        Y_max: Most recent year in range (2024)
        T_old: Tolerance for oldest years (20)
        T_recent: Tolerance for most recent years (5)
        I_old: Intolerance for oldest years (50)
        I_recent: Intolerance for most recent years (15)

    Returns:
        T: Tolerance threshold for this year
        I: Intolerance threshold for this year
    """
    # Linear interpolation based on year position
    ratio = (Y_max - year) / (Y_max - Y_min)
    T = T_recent + (T_old - T_recent) * ratio
    I = I_recent + (I_old - I_recent) * ratio
    return T, I


def calculate_TAI(prediction_year, groundtruth_year, Y_min=1700, Y_max=2024,
                  T_old=20, T_recent=5, I_old=50, I_recent=15):
    """
    Calculate Time Awareness Index (TAI) for a single prediction.

    Equation 4 from the paper:
        TAI = 1                          if |y_pred - y_GT| <= T(y)
        TAI = 1 - |y_pred - y_GT| / I(y)  if T(y) < |y_pred - y_GT| < I(y)
        TAI = 0                          if |y_pred - y_GT| >= I(y)

    Args:
        prediction_year: Predicted year
        groundtruth_year: Ground truth year
        Y_min, Y_max: Year range boundaries
        T_old, T_recent: Tolerance range (20 for old, 5 for recent)
        I_old, I_recent: Intolerance range (50 for old, 15 for recent)

    Returns:
        TAI score in [0, 1]
    """
    T, I = calculate_dynamic_thresholds(
        groundtruth_year, Y_min, Y_max, T_old, T_recent, I_old, I_recent
    )

    difference = abs(prediction_year - groundtruth_year)

    if difference <= T:
        return 1.0
    elif T < difference < I:
        return 1.0 - (difference / I)
    else:
        return 0.0


def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error between true and predicted years."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def evaluate_chronological_order(actual_labels, ascending=True):
    """
    Evaluate how well a sequence maintains chronological order using
    bubble sort distance (inversion count).

    Args:
        actual_labels: List of years in predicted order
        ascending: Whether expected order is ascending

    Returns:
        num_swaps: Number of adjacent swaps needed to sort
        normalized_distance: Normalized swap distance in [0, 1]
    """
    sorted_target = sorted(actual_labels) if ascending else sorted(actual_labels, reverse=True)
    num_swaps = count_adjacent_swaps(sorted_target, actual_labels)
    max_swaps = len(actual_labels) * (len(actual_labels) - 1) // 2
    normalized_distance = num_swaps / max_swaps if max_swaps > 0 else 0
    return num_swaps, normalized_distance


def count_adjacent_swaps(sorted_labels, actual_labels):
    """Count inversions (adjacent swaps needed to sort)."""
    label_to_index = {label: i for i, label in enumerate(sorted_labels)}
    indices = [label_to_index[label] for label in actual_labels]

    swap_count = 0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if indices[i] > indices[j]:
                swap_count += 1
    return swap_count


def compute_ranking_metrics(predicted_order, true_order):
    """
    Compute all ranking metrics from Section 5.4 of the paper.

    Metrics:
    - Spearman's rho: rank correlation
    - Kendall's tau: concordance measure
    - delta_MNDL: Modified Normalised Damerau-Levenshtein Distance
      delta_MNDL = 1 - 2 * S / M, where S = inversion count, M = N(N-1)/2

    All metrics range from -1 to 1, where 1 = perfect chronological order,
    -1 = perfectly reversed (both considered structurally correct).

    Args:
        predicted_order: Predicted ordering (array of indices or values)
        true_order: True ordering

    Returns:
        Dictionary with spearman_rho, kendall_tau, delta_MNDL
    """
    rho, _ = spearmanr(predicted_order, true_order)
    tau, _ = kendalltau(predicted_order, true_order)

    # delta_MNDL: 1 - 2*S/M where S is inversion count, M = N(N-1)/2
    ordered_items = [true_order[i] for i in np.argsort(predicted_order)]
    num_swaps, _ = evaluate_chronological_order(list(ordered_items))
    N = len(predicted_order)
    M = N * (N - 1) // 2
    delta_MNDL = 1.0 - 2.0 * num_swaps / M if M > 0 else 1.0

    return {
        'spearman_rho': float(rho) if not np.isnan(rho) else 0.0,
        'kendall_tau': float(tau) if not np.isnan(tau) else 0.0,
        'delta_MNDL': float(delta_MNDL),
    }


def calculate_mae_per_decade(predictions, ground_truths):
    """
    Calculate MAE per decade for detailed error analysis.

    Returns:
        Dictionary mapping decade (int) to MAE (float)
    """
    from collections import defaultdict

    decade_errors = defaultdict(list)
    for pred, actual in zip(predictions, ground_truths):
        decade = (actual // 10) * 10
        decade_errors[decade].append(abs(pred - actual))

    return {decade: float(np.mean(errors))
            for decade, errors in sorted(decade_errors.items())}


def calculate_mae_per_class(predictions, ground_truths, classes):
    """
    Calculate MAE and TAI per object class (Table 3 in paper).

    Args:
        predictions: List of predicted years
        ground_truths: List of ground truth years
        classes: List of class labels for each sample

    Returns:
        Dictionary mapping class name to {mae, tai, count, year_range}
    """
    from collections import defaultdict

    class_data = defaultdict(lambda: {'preds': [], 'gts': []})
    for pred, gt, cls in zip(predictions, ground_truths, classes):
        class_data[cls]['preds'].append(pred)
        class_data[cls]['gts'].append(gt)

    results = {}
    for cls, data in sorted(class_data.items()):
        preds = np.array(data['preds'])
        gts = np.array(data['gts'])
        mae = float(np.mean(np.abs(preds - gts)))
        tai = float(np.mean([calculate_TAI(p, g) for p, g in zip(preds, gts)]))
        results[cls] = {
            'mae': mae,
            'tai': tai,
            'count': len(preds),
            'year_range': (int(gts.min()), int(gts.max())),
        }
    return results


def print_evaluation_summary(predictions, ground_truths, model_name="Model",
                             classes=None):
    """
    Print a comprehensive evaluation summary matching paper metrics.

    Args:
        predictions: List of predicted years
        ground_truths: List of ground truth years
        model_name: Name of the model for display
        classes: Optional list of class labels per sample
    """
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    mae = mean_absolute_error(ground_truths, predictions)
    tai_scores = [calculate_TAI(p, g) for p, g in zip(predictions, ground_truths)]
    mean_tai = float(np.mean(tai_scores))
    errors = np.abs(predictions - ground_truths)

    print(f"\n{'=' * 60}")
    print(f"Evaluation Summary: {model_name}")
    print(f"{'=' * 60}")
    print(f"Samples: {len(predictions)}")
    print(f"MAE:     {mae:.2f} years")
    print(f"TAI:     {mean_tai:.3f}")

    print(f"\nError distribution:")
    print(f"  Median: {np.median(errors):.0f}  Std: {np.std(errors):.1f}")
    for threshold in [5, 10, 20, 50]:
        acc = np.mean(errors <= threshold) * 100
        print(f"  Within {threshold:>2} years: {acc:.1f}%")

    if classes is not None:
        print(f"\nPer-class results:")
        class_results = calculate_mae_per_class(predictions, ground_truths, classes)
        print(f"  {'Class':<25} {'MAE':>6} {'TAI':>6} {'Count':>6} {'Range'}")
        print(f"  {'-' * 65}")
        for cls, res in class_results.items():
            yr = f"[{res['year_range'][0]},{res['year_range'][1]}]"
            print(f"  {cls:<25} {res['mae']:>6.2f} {res['tai']:>6.2f} "
                  f"{res['count']:>6} {yr}")

    print(f"{'=' * 60}\n")
