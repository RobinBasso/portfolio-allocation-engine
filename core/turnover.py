import numpy as np


def calculate_turnover(
    previous_weights: np.ndarray,
    target_weights: np.ndarray,
) -> float:
    """Calculate one-way portfolio turnover."""

    previous_weights = np.asarray(previous_weights, dtype=float)
    target_weights = np.asarray(target_weights, dtype=float)

    if previous_weights.ndim != 1 or target_weights.ndim != 1:
        raise ValueError("Weights must be one-dimensional.")

    if previous_weights.shape != target_weights.shape:
        raise ValueError("Weight arrays must have the same shape.")

    if not np.isfinite(previous_weights).all():
        raise ValueError("Weights must be finite.")

    if not np.isfinite(target_weights).all():
        raise ValueError("Weights must be finite.")

    if (previous_weights < 0).any() or (target_weights < 0).any():
        raise ValueError("Weights must be non-negative.")

    if not np.isclose(previous_weights.sum(), 1.0):
        raise ValueError("Previous weights must sum to one.")

    if not np.isclose(target_weights.sum(), 1.0):
        raise ValueError("Target weights must sum to one.")

    return float(0.5 * np.abs(target_weights - previous_weights).sum())
