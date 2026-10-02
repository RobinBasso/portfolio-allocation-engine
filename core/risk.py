import numpy as np
import pandas as pd


def validate_weights(
    returns: pd.DataFrame,
    weights: np.ndarray,
) -> np.ndarray:
    """Validate and return portfolio weights as a NumPy array."""
    weights = np.asarray(weights, dtype=float)

    if len(weights) != returns.shape[1]:
        raise ValueError(
            "Number of weights must match number of assets."
        )

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    return weights


def portfolio_volatility(
    covariance: pd.DataFrame,
    weights: np.ndarray,
) -> float:
    """
    Calculate portfolio volatility from a covariance matrix.

    Formula:
        sigma_p = sqrt(w.T @ Sigma @ w)
    """
    covariance_matrix = covariance.to_numpy()
    weights = np.asarray(weights, dtype=float)

    variance = weights @ covariance_matrix @ weights

    return float(np.sqrt(variance))


def marginal_contribution_to_risk(
    covariance: pd.DataFrame,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Calculate each asset's marginal contribution to portfolio risk.

    Formula:
        MCR_i = (Sigma @ w)_i / sigma_p
    """
    covariance_matrix = covariance.to_numpy()
    weights = np.asarray(weights, dtype=float)

    portfolio_risk = portfolio_volatility(
        covariance,
        weights,
    )

    if portfolio_risk <= 0:
        raise ValueError("Portfolio volatility must be positive.")

    covariance_times_weights = covariance_matrix @ weights

    return covariance_times_weights / portfolio_risk


def component_contribution_to_risk(
    covariance: pd.DataFrame,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Calculate each asset's absolute contribution to portfolio risk.

    Formula:
        CR_i = w_i * MCR_i
    """
    weights = np.asarray(weights, dtype=float)

    marginal_risk = marginal_contribution_to_risk(
        covariance,
        weights,
    )

    return weights * marginal_risk


def percentage_contribution_to_risk(
    covariance: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """
    Calculate each asset's percentage contribution to portfolio risk.
    """
    component_risk = component_contribution_to_risk(
        covariance,
        weights,
    )

    portfolio_risk = portfolio_volatility(
        covariance,
        weights,
    )

    contribution = component_risk / portfolio_risk

    return pd.Series(
        contribution,
        index=covariance.index,
        name="risk_contribution",
    )