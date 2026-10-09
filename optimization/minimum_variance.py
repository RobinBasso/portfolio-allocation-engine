import numpy as np
import pandas as pd
from scipy.optimize import minimize


def minimum_variance_weights(
    covariance: pd.DataFrame,
    max_weight: float = 0.40,
) -> pd.Series:
    """
    Find the fully invested, long-only portfolio
    with minimum variance under individual weight limits.
    """
    if covariance.empty:
        raise ValueError("Covariance matrix cannot be empty.")

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    if not covariance.index.equals(covariance.columns):
        raise ValueError(
            "Covariance row and column labels must match."
        )

    if not np.isfinite(covariance.to_numpy()).all():
        raise ValueError("Covariance matrix must be finite.")

    if not np.allclose(
        covariance.to_numpy(),
        covariance.to_numpy().T,
    ):
        raise ValueError("Covariance matrix must be symmetric.")

    if max_weight <= 0 or max_weight > 1:
        raise ValueError("max_weight must be in (0, 1].")

    n_assets = len(covariance)

    if n_assets * max_weight < 1.0 - 1e-10:
        raise ValueError(
            "The maximum weight is too restrictive "
            "to fully invest the portfolio."
        )

    covariance_matrix = covariance.to_numpy(dtype=float)

    # Minimize variance; the factor 1/2 does not change the optimum.
    def objective(weights: np.ndarray) -> float:
        return float(
            0.5 * weights @ covariance_matrix @ weights
        )

    initial_weights = np.full(n_assets, 1.0 / n_assets)

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n_assets,
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: weights.sum() - 1.0,
            }
        ],
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(
            f"Optimization failed: {result.message}"
        )

    weights = result.x

    if not np.isclose(weights.sum(), 1.0, atol=1e-7):
        raise RuntimeError("Optimized weights do not sum to one.")

    if np.any(weights < -1e-7) or np.any(
        weights > max_weight + 1e-7
    ):
        raise RuntimeError("Optimized weights violate constraints.")

    weights = np.clip(weights, 0.0, max_weight)
    weights /= weights.sum()

    return pd.Series(
        weights,
        index=covariance.columns,
        name="weight",
    )