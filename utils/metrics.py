"""
Evaluation metrics for temporal awareness assessment
Includes TAI (Time Awareness Index) and chronological ordering metrics
"""

import numpy as np
from scipy.stats import spearmanr, kendalltau


def calculate_dynamic_thresholds(year, Y_min=1700, Y_max=2024, T_old=20, T_recent=4, I_old=50, I_recent=10):
    """
    Calculate dynamic tolerance and intolerance thresholds based on the year.
    
    Args:
        year: The ground truth year
        Y_min: Earliest year in range
        Y_max: Most recent year in range
        T_old: Tolerance for oldest years
        T_recent: Tolerance for recent years
        I_old: Intolerance for oldest years
        I_recent: Intolerance for recent years
    
    Returns:
        T: Tolerance threshold
        I: Intolerance threshold
    """
    T = T_recent + (T_old - T_recent) * (Y_max - year) / (Y_max - Y_min)
    I = I_recent + (I_old - I_recent) * (Y_max - year) / (Y_max - Y_min)
    return T, I


def calculate_TAI(prediction_year, groundtruth_year, Y_min=1700, Y_max=2024, 
                 T_old=20, T_recent=4, I_old=50, I_recent=10):
    """
    Calculate Time Awareness Index (TAI) for a single prediction.
    
    TAI applies adaptive tolerance based on the historical period:
    - Older years get more tolerance (reflecting historical uncertainty)
    - Recent years require stricter accuracy
    
    Args:
        prediction_year: The predicted year
        groundtruth_year: The actual (ground truth) year
        Y_min: Earliest year in the range
        Y_max: Most recent year in the range
        T_old: Tolerance for oldest years
        T_recent: Tolerance for recent years
        I_old: Intolerance for oldest years
        I_recent: Intolerance for recent years
    
    Returns:
        TAI score between 0 and 1
    """
    # Calculate dynamic thresholds based on the ground truth year
    T, I = calculate_dynamic_thresholds(groundtruth_year, Y_min, Y_max, 
                                      T_old, T_recent, I_old, I_recent)
    
    # Calculate absolute difference between prediction and ground truth
    difference = abs(prediction_year - groundtruth_year)
    
    # Apply TAI calculation rules
    if difference <= T:
        return 1.0
    elif T < difference < I:
        return 1.0 - (difference / I)
    else:
        return 0.0


def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def evaluate_chronological_order(actual_labels, ascending=True):
    """
    Evaluate how well a sequence maintains chronological order.
    
    Args:
        actual_labels: List of years in the order they appear
        ascending: Whether to check for ascending (True) or descending (False) order
    
    Returns:
        num_swaps: Number of adjacent swaps needed to sort
        normalized_distance: Normalized swap distance (0 to 1, where 0 is perfect)
    """
    sorted_labels_target = sorted(actual_labels) if ascending else sorted(actual_labels, reverse=True)
    num_swaps = count_adjacent_swaps(sorted_labels_target, actual_labels)
    max_swaps = len(actual_labels) * (len(actual_labels) - 1) // 2
    normalized_distance = num_swaps / max_swaps if max_swaps > 0 else 0
    return num_swaps, normalized_distance


def count_adjacent_swaps(sorted_labels, actual_labels):
    """Count the number of adjacent swaps needed to sort the list"""
    swap_count = 0
    label_to_index = {label: i for i, label in enumerate(sorted_labels)}
    indices = [label_to_index[label] for label in actual_labels]
    
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if indices[i] > indices[j]:
                swap_count += 1
    
    return swap_count


def compute_ranking_metrics(predicted_order, true_order):
    """
    Compute multiple ranking correlation metrics.
    
    Args:
        predicted_order: Predicted ordering of items
        true_order: True ordering of items
    
    Returns:
        Dictionary with Spearman's rho, Kendall's tau, and normalized swap distance
    """
    # Spearman's rank correlation
    rho, _ = spearmanr(predicted_order, true_order)
    
    # Kendall's tau
    tau, _ = kendalltau(predicted_order, true_order)
    
    # Normalized swap distance
    ordered_items = [true_order[i] for i in np.argsort(predicted_order)]
    _, norm_distance = evaluate_chronological_order(ordered_items)
    # Convert to correlation-like metric (1 = perfect, -1 = reversed)
    norm_distance = 1 - 2 * norm_distance
    
    return {
        'spearman_rho': rho,
        'kendall_tau': tau,
        'normalized_swap_distance': norm_distance
    }


def calculate_mae_per_decade(predictions, ground_truths):
    """
    Calculate MAE per decade for detailed error analysis.
    
    Args:
        predictions: List of predicted years
        ground_truths: List of ground truth years
    
    Returns:
        Dictionary mapping decades to their MAE values
    """
    from collections import defaultdict
    
    decade_errors = defaultdict(list)
    
    for pred, actual in zip(predictions, ground_truths):
        decade = (actual // 10) * 10  # Group by decade
        decade_errors[decade].append(abs(pred - actual))
    
    mae_per_decade = {}
    for decade, errors in sorted(decade_errors.items()):
        mae_per_decade[decade] = np.mean(errors) if errors else 0
    
    return mae_per_decade


def print_evaluation_summary(predictions, ground_truths, model_name="Model"):
    """
    Print a comprehensive evaluation summary.
    
    Args:
        predictions: List of predicted years
        ground_truths: List of ground truth years
        model_name: Name of the model for display
    """
    # Calculate metrics
    mae = mean_absolute_error(ground_truths, predictions)
    tai_scores = [calculate_TAI(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    mean_tai = np.mean(tai_scores)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Evaluation Summary for {model_name}")
    print(f"{'='*50}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} years")
    print(f"Time Awareness Index (TAI): {mean_tai:.3f}")
    
    # Additional statistics
    errors = np.abs(np.array(predictions) - np.array(ground_truths))
    print(f"\nError Statistics:")
    print(f"  Min error: {np.min(errors):.0f} years")
    print(f"  Max error: {np.max(errors):.0f} years")
    print(f"  Median error: {np.median(errors):.0f} years")
    print(f"  Std deviation: {np.std(errors):.1f} years")
    
    # Accuracy at different thresholds
    print(f"\nAccuracy at different thresholds:")
    for threshold in [5, 10, 20, 50]:
        accuracy = np.mean(errors <= threshold) * 100
        print(f"  Within {threshold} years: {accuracy:.1f}%")
    
    print(f"{'='*50}\n")