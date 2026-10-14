
import numpy as np
import pandas as pd

from core.covariance import calculate_covariance
from core.portfolio import annualized_return, sharpe_ratio
from core.risk import (
    percentage_contribution_to_risk,
    portfolio_volatility,
)
from core.returns import calculate_returns


def build_allocation_report(
    prices: pd.DataFrame,
    weights: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Build portfolio metrics and asset-level risk attribution."""

    if prices.empty:
        raise ValueError("Price data cannot be empty.")
    if not isinstance(weights, pd.Series):
        raise TypeError("Weights must be a pandas Series indexed by asset.")
    if not weights.index.is_unique:
        raise ValueError("Weight asset labels must be unique.")
    if set(weights.index) != set(prices.columns):
        raise ValueError("Weight labels must match price columns.")

    weights = weights.reindex(prices.columns).astype(float)
    weight_values = weights.to_numpy()

    if not np.isfinite(weight_values).all():
        raise ValueError("Weights must be finite.")
    if np.any(weight_values < 0):
        raise ValueError("Weights must be non-negative.")
    if not np.isclose(weight_values.sum(), 1.0):
        raise ValueError("Weights must sum to one.")

    returns = calculate_returns(prices)
    covariance = calculate_covariance(returns)

    portfolio_returns = returns @ weight_values
    annual_return = annualized_return(portfolio_returns)
    annual_volatility = portfolio_volatility(
        covariance,
        weight_values,
    )

    risk_contributions = percentage_contribution_to_risk(
        covariance,
        weight_values,
    )

    allocation_report = pd.DataFrame(
        {
            "Allocation": weights,
            "Risk Contribution": risk_contributions,
        }
    )
    allocation_report["Risk-to-Weight Ratio"] = (
        allocation_report["Risk Contribution"]
        / allocation_report["Allocation"].replace(0.0, np.nan)
    )

    portfolio_metrics = pd.Series(
        {
            "Annualized Return": annual_return,
            "Annualized Volatility": annual_volatility,
            "Sharpe Ratio": sharpe_ratio(
                annual_return,
                annual_volatility,
                risk_free_rate,
            ),
        },
        name="Portfolio",
    )

    return {
        "portfolio_metrics": portfolio_metrics,
        "asset_report": allocation_report,
    }
