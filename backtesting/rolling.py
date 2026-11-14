
import numpy as np
import pandas as pd

from core.covariance import (
    calculate_covariance,
    calculate_shrinkage_covariance,
    calculate_ledoit_wolf_covariance,
)
from core.returns import calculate_returns
from core.turnover import calculate_turnover
from optimization.minimum_variance import minimum_variance_weights


COVARIANCE_ESTIMATORS = {
    "sample": calculate_covariance,
    "shrinkage": calculate_shrinkage_covariance,
    "ledoit_wolf": calculate_ledoit_wolf_covariance,
}


def _buy_and_hold_returns(
    asset_returns: pd.DataFrame,
    initial_weights: np.ndarray,
) -> pd.Series:
    """Calculate daily portfolio returns without intra-period rebalancing."""
    asset_growth = (1 + asset_returns).cumprod()
    portfolio_value = asset_growth @ initial_weights
    previous_value = portfolio_value.shift(1, fill_value=1.0)
    return (portfolio_value / previous_value - 1).rename("Portfolio")


def _drift_weights(
    initial_weights: np.ndarray,
    asset_returns: pd.DataFrame,
) -> np.ndarray:
    """Calculate portfolio weights after buy-and-hold returns."""
    asset_growth = (1 + asset_returns).prod(axis=0).to_numpy()
    ending_values = initial_weights * asset_growth
    return ending_values / ending_values.sum()


def _apply_transaction_cost(
    gross_returns: pd.Series,
    turnover: float,
    transaction_cost_bps: float,
) -> pd.Series:
    """Deduct transaction costs from the first return of a period."""
    net_returns = gross_returns.copy()
    cost_rate = transaction_cost_bps / 10_000
    cost_fraction = turnover * cost_rate

    if len(net_returns) > 0:
        first_return = net_returns.iloc[0]
        net_returns.iloc[0] = (
            (1 + first_return) * (1 - cost_fraction) - 1
        )

    return net_returns


def run_rolling_backtest(
    prices: pd.DataFrame,
    estimation_window: int = 252,
    rebalance_frequency: int = 21,
    max_weight: float = 0.40,
    transaction_cost_bps: float = 10.0,
    covariance_method: str = "sample",
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

    if (
        not isinstance(transaction_cost_bps, (int, float))
        or isinstance(transaction_cost_bps, bool)
        or not np.isfinite(transaction_cost_bps)
        or transaction_cost_bps < 0
        or transaction_cost_bps >= 10_000
    ):
        raise ValueError(
            "Transaction cost must be finite and between 0 and 10,000 bps."
        )

    if covariance_method not in COVARIANCE_ESTIMATORS:
        raise ValueError(
            f"Unknown covariance method: {covariance_method}. "
            f"Choose from {list(COVARIANCE_ESTIMATORS)}."
        )

    if prices.shape[1] < 2:
        raise ValueError("At least two assets are required.")

    if len(prices) <= estimation_window:
        raise ValueError("Insufficient prices for an out-of-sample period.")

    equal_weights = np.full(prices.shape[1], 1.0 / prices.shape[1])
    previous_equal_weights = equal_weights.copy()
    previous_optimized_weights = None

    results = []
    train_end = estimation_window

    covariance_estimator = COVARIANCE_ESTIMATORS[covariance_method]

    while train_end < len(prices):
        test_end = min(train_end + rebalance_frequency, len(prices))

        training_prices = prices.iloc[
            train_end - estimation_window:train_end
        ]

        training_returns = calculate_returns(training_prices)

        covariance = covariance_estimator(
            training_returns,
            annualize=False,
        )

        optimized_weights = minimum_variance_weights(
            covariance,
            max_weight=max_weight,
        ).to_numpy()

        test_prices = prices.iloc[train_end - 1:test_end]
        test_returns = calculate_returns(test_prices)

        if not test_returns.columns.equals(training_returns.columns):
            raise ValueError("Asset order mismatch between prices and weights.")

        if previous_optimized_weights is None:
            equal_turnover = 0.0
            optimized_turnover = 0.0
        else:
            equal_turnover = calculate_turnover(
                previous_equal_weights,
                equal_weights,
            )
            optimized_turnover = calculate_turnover(
                previous_optimized_weights,
                optimized_weights,
            )

        equal_returns = _buy_and_hold_returns(
            test_returns,
            equal_weights,
        )

        optimized_returns = _buy_and_hold_returns(
            test_returns,
            optimized_weights,
        )

        equal_net_returns = _apply_transaction_cost(
            equal_returns,
            equal_turnover,
            transaction_cost_bps,
        )

        optimized_net_returns = _apply_transaction_cost(
            optimized_returns,
            optimized_turnover,
            transaction_cost_bps,
        )

        period_results = pd.DataFrame(
            {
                "Equal Weight": equal_returns,
                "Minimum Variance": optimized_returns,
                "Equal Weight Net": equal_net_returns,
                "Minimum Variance Net": optimized_net_returns,
                "Equal Weight Turnover": 0.0,
                "Minimum Variance Turnover": 0.0,
            },
            index=test_returns.index,
        )

        period_results.iloc[
            0, period_results.columns.get_loc("Equal Weight Turnover")
        ] = equal_turnover

        period_results.iloc[
            0, period_results.columns.get_loc("Minimum Variance Turnover")
        ] = optimized_turnover

        results.append(period_results)

        previous_equal_weights = _drift_weights(
            equal_weights,
            test_returns,
        )

        previous_optimized_weights = _drift_weights(
            optimized_weights,
            test_returns,
        )

        train_end += rebalance_frequency

    return pd.concat(results).rename_axis("Date")
