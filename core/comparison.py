
import numpy as np
import pandas as pd

from core.covariance import calculate_covariance
from core.portfolio import annualized_return, sharpe_ratio
from core.risk import portfolio_volatility
from core.returns import calculate_returns
from optimization.minimum_variance import minimum_variance_weights


def compare_portfolios(prices: pd.DataFrame) -> pd.DataFrame:
    """Compare equal-weight and minimum-variance portfolios."""

    if prices.empty:
        raise ValueError("Price data cannot be empty.")

    returns = calculate_returns(prices)
    n_assets = returns.shape[1]

    if n_assets == 0:
        raise ValueError("Price data must contain at least one asset.")

    covariance = calculate_covariance(returns)

    equal_weights = np.full(n_assets, 1.0 / n_assets)
    optimized_weights = minimum_variance_weights(covariance).to_numpy()

    portfolios = {
        "Equal Weight": equal_weights,
        "Minimum Variance": optimized_weights,
    }

    results = {}

    for name, weights in portfolios.items():
        daily_portfolio_returns = returns @ weights

        annual_return = annualized_return(daily_portfolio_returns)
        annual_volatility = portfolio_volatility(covariance, weights)

        results[name] = {
            "Annualized Return": annual_return,
            "Annualized Volatility": annual_volatility,
            "Sharpe Ratio": sharpe_ratio(
                annual_return,
                annual_volatility,
            ),
        }

    return pd.DataFrame(results).T