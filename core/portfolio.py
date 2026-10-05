import numpy as np
import pandas as pd

from core.covariance import calculate_covariance
from core.returns import calculate_returns
from core.risk import (
    percentage_contribution_to_risk,
    portfolio_volatility,
)


TRADING_DAYS_PER_YEAR = 252


def portfolio_return(
    returns: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """Calculate the portfolio return series."""
    weights = np.asarray(weights, dtype=float)

    if len(weights) != returns.shape[1]:
        raise ValueError(
            "Number of weights must match number of assets."
        )

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
    """Calculate annualized volatility from return observations."""
    return (
        portfolio_returns.std(ddof=1)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def analyze_portfolio(
    prices: pd.DataFrame,
    weights: np.ndarray,
) -> dict:
    """
    Calculate the main analytics for a portfolio.

    Returns
    -------
    dict
        Portfolio return, volatility, covariance matrix,
        and risk contribution data.
    """
    returns = calculate_returns(prices)

    weights = np.asarray(weights, dtype=float)

    if len(weights) != returns.shape[1]:
        raise ValueError(
            "Number of weights must match number of assets."
        )

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    covariance = calculate_covariance(returns)

    portfolio_returns = portfolio_return(
        returns,
        weights,
    )

    volatility = portfolio_volatility(
        covariance,
        weights,
    )

    risk_contributions = percentage_contribution_to_risk(
        covariance,
        weights,
    )

    return {
        "returns": returns,
        "portfolio_returns": portfolio_returns,
        "annualized_return": annualized_return(
            portfolio_returns
        ),
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe_ratio(
            annualized_return(portfolio_returns),
            volatility,
        ),
        "covariance": covariance,
        "risk_contributions": risk_contributions,
    }


def sharpe_ratio(
    annualized_return_value: float,
    annualized_volatility_value: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate the annualized Sharpe ratio."""
    if annualized_volatility_value <= 0:
        raise ValueError(
            "Annualized volatility must be positive."
        )

    return (
        annualized_return_value - risk_free_rate
    ) / annualized_volatility_value