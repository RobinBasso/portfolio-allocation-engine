import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def portfolio_return(
    returns: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """Calculate the portfolio return series."""
    weights = np.asarray(weights)

    if len(weights) != returns.shape[1]:
        raise ValueError("Number of weights must match number of assets.")

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    return returns @ weights


def annualized_return(
    portfolio_returns: pd.Series,
) -> float:
    """Calculate annualized arithmetic return."""
    return portfolio_returns.mean() * TRADING_DAYS_PER_YEAR


def annualized_volatility(
    portfolio_returns: pd.Series,
) -> float:
    """Calculate annualized portfolio volatility."""
    return (
        portfolio_returns.std(ddof=1)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )