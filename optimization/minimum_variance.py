
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def minimum_variance_weights(
    covariance: pd.DataFrame,
    max_weight: float | dict[str, float] = 0.40,
) -> pd.Series:
    """Find long-only minimum-variance weights under allocation caps."""

    if covariance.empty:
        raise ValueError("Covariance matrix cannot be empty.")
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if not covariance.index.equals(covariance.columns):
        raise ValueError("Covariance row and column labels must match.")
    if not np.isfinite(covariance.to_numpy()).all():
        raise ValueError("Covariance matrix must be finite.")
    if not np.allclose(covariance.to_numpy(), covariance.to_numpy().T):
        raise ValueError("Covariance matrix must be symmetric.")

    n_assets = len(covariance)
    asset_names = list(covariance.columns)

    if isinstance(max_weight, dict):
        unknown_assets = set(max_weight) - set(asset_names)
        if unknown_assets:
            raise ValueError(f"Unknown assets in weight limits: {unknown_assets}")

        missing_assets = set(asset_names) - set(max_weight)
        if missing_assets:
            raise ValueError(f"Missing weight limits for assets: {missing_assets}")

        upper_bounds = np.array(
            [max_weight[asset] for asset in asset_names],
            dtype=float,
        )
    else:
        upper_bounds = np.full(n_assets, max_weight, dtype=float)

    if not np.isfinite(upper_bounds).all():
        raise ValueError("Weight limits must be finite.")
    if np.any(upper_bounds <= 0) or np.any(upper_bounds > 1):
        raise ValueError("Weight limits must be in (0, 1].")
    if upper_bounds.sum() < 1.0 - 1e-10:
        raise ValueError("Weight limits are too restrictive to fully invest.")

    covariance_matrix = covariance.to_numpy(dtype=float)

    def objective(weights: np.ndarray) -> float:
        return float(0.5 * weights @ covariance_matrix @ weights)

    initial_weights = upper_bounds / upper_bounds.sum()

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=[(0.0, float(limit)) for limit in upper_bounds],
        constraints=[
            {"type": "eq", "fun": lambda weights: weights.sum() - 1.0}
        ],
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    weights = result.x

    if not np.isclose(weights.sum(), 1.0, atol=1e-7):
        raise RuntimeError("Optimized weights do not sum to one.")
    if np.any(weights < -1e-7) or np.any(weights > upper_bounds + 1e-7):
        raise RuntimeError("Optimized weights violate constraints.")

    weights = np.clip(weights, 0.0, upper_bounds)
    weights /= weights.sum()

    return pd.Series(weights, index=covariance.columns, name="weight")
