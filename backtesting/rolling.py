import numpy as np
import pandas as pd

from core.covariance import calculate_covariance
from core.returns import calculate_returns
from optimization.minimum_variance import minimum_variance_weights


def _buy_and_hold_returns(
    asset_returns: pd.DataFrame,
    initial_weights: np.ndarray,
) -> pd.Series:
    """Calculate daily portfolio returns without intra-period rebalancing."""

    asset_growth = (1 + asset_returns).cumprod()

    portfolio_value = asset_growth @ initial_weights
    previous_value = portfolio_value.shift(1, fill_value=1.0)

    return (portfolio_value / previous_value - 1).rename("Portfolio")


def run_rolling_backtest(
    prices: pd.DataFrame,
    estimation_window: int = 252,
    rebalance_frequency: int = 21,
    max_weight: float = 0.40,
) -> pd.DataFrame:
    """Run a chronological rolling-window portfolio backtest."""

    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Prices must be a non-empty DataFrame.")

    if not prices.columns.is_unique:
        raise ValueError("Asset columns must be unique.")

    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price index must be unique and ascending.")

    if not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError("Prices must be finite.")

    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")

    if (
        not isinstance(estimation_window, int)
        or isinstance(estimation_window, bool)
        or estimation_window < 2
    ):
        raise ValueError("Estimation window must be an integer of at least 2.")

    if (
        not isinstance(rebalance_frequency, int)
        or isinstance(rebalance_frequency, bool)
        or rebalance_frequency < 1
    ):
        raise ValueError("Rebalancing frequency must be a positive integer.")

    if prices.shape[1] < 2:
        raise ValueError("At least two assets are required.")

    if len(prices) <= estimation_window:
        raise ValueError("Insufficient prices for an out-of-sample period.")

    equal_weights = np.full(prices.shape[1], 1.0 / prices.shape[1])
    results = []
    train_end = estimation_window

    while train_end < len(prices):
        test_end = min(train_end + rebalance_frequency, len(prices))

        training_prices = prices.iloc[
            train_end - estimation_window:train_end
        ]
        training_returns = calculate_returns(training_prices)

        covariance = calculate_covariance(
            training_returns,
            annualize=False,
        )

        optimized_weights = minimum_variance_weights(
            covariance,
            max_weight=max_weight,
        )

        test_prices = prices.iloc[train_end - 1:test_end]
        test_returns = calculate_returns(test_prices)

        if not test_returns.columns.equals(optimized_weights.index):
            raise ValueError("Asset order mismatch between prices and weights.")

        equal_returns = _buy_and_hold_returns(
            test_returns,
            equal_weights,
        )
        optimized_returns = _buy_and_hold_returns(
            test_returns,
            optimized_weights.to_numpy(),
        )

        results.append(pd.DataFrame(
            {
                "Equal Weight": equal_returns,
                "Minimum Variance": optimized_returns,
            },
            index=test_returns.index,
        ))

        train_end += rebalance_frequency

    return pd.concat(results).rename_axis("Date")