
import numpy as np
import pandas as pd

from backtesting.rolling import run_rolling_backtest
from backtesting.weights import collect_rolling_weights
from core.returns import calculate_returns


def calculate_daily_attribution(
    asset_returns: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    transaction_costs: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Calculate daily asset-level return contributions.

    Weights represent beginning-of-day portfolio weights.
    Transaction costs are expressed as deductions from daily returns.
    """

    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns must be a pandas DataFrame.")

    if not isinstance(portfolio_weights, pd.DataFrame):
        raise TypeError("portfolio_weights must be a pandas DataFrame.")

    if asset_returns.empty or asset_returns.shape[1] == 0:
        raise ValueError("asset_returns must not be empty.")

    if not asset_returns.columns.is_unique:
        raise ValueError("Asset columns must be unique.")

    if not asset_returns.index.is_unique:
        raise ValueError("Return dates must be unique.")

    if not portfolio_weights.columns.is_unique:
        raise ValueError("Weight columns must be unique.")

    if not portfolio_weights.index.is_unique:
        raise ValueError("Weight dates must be unique.")

    if not asset_returns.index.equals(portfolio_weights.index):
        raise ValueError("Return and weight dates must match exactly.")

    if not asset_returns.columns.equals(portfolio_weights.columns):
        raise ValueError("Return and weight assets must match in the same order.")

    reserved_columns = {
        "Gross Portfolio Return",
        "Transaction Cost",
        "Net Portfolio Return",
    }

    if reserved_columns.intersection(asset_returns.columns):
        raise ValueError("Asset names conflict with reserved output columns.")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in asset_returns.dtypes
    ):
        raise TypeError("Asset returns must be numeric.")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in portfolio_weights.dtypes
    ):
        raise TypeError("Portfolio weights must be numeric.")

    returns = asset_returns.to_numpy(dtype=float)
    weights = portfolio_weights.to_numpy(dtype=float)

    if not np.isfinite(returns).all():
        raise ValueError("Asset returns must contain only finite values.")

    if not np.isfinite(weights).all():
        raise ValueError("Portfolio weights must contain only finite values.")

    if (returns <= -1.0).any():
        raise ValueError("Asset returns must be greater than -100%.")

    if (weights < 0.0).any():
        raise ValueError("Portfolio weights must be non-negative.")

    if not np.allclose(weights.sum(axis=1), 1.0):
        raise ValueError("Portfolio weights must sum to 1 on every date.")

    if transaction_costs is None:
        costs = pd.Series(0.0, index=asset_returns.index)
    else:
        if not isinstance(transaction_costs, pd.Series):
            raise TypeError("transaction_costs must be a pandas Series.")

        if not transaction_costs.index.equals(asset_returns.index):
            raise ValueError(
                "Transaction cost dates must match return dates exactly."
            )

        if not pd.api.types.is_numeric_dtype(transaction_costs.dtype):
            raise TypeError("Transaction costs must be numeric.")

        costs = transaction_costs.astype(float)

        if not np.isfinite(costs.to_numpy()).all():
            raise ValueError("Transaction costs must be finite.")

        if (costs < 0.0).any():
            raise ValueError("Transaction costs must be non-negative.")

    contributions = asset_returns * portfolio_weights
    gross_returns = contributions.sum(axis=1)

    attribution = contributions.copy()
    attribution["Gross Portfolio Return"] = gross_returns
    attribution["Transaction Cost"] = costs
    attribution["Net Portfolio Return"] = gross_returns - costs

    return attribution


def _build_daily_weights(
    asset_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct beginning-of-day weights from periodic target weights.

    Weights drift daily between rebalancing dates and reset to the
    optimiser's target weights at each new period.
    """
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns must be a pandas DataFrame.")

    if not isinstance(target_weights, pd.DataFrame):
        raise TypeError("target_weights must be a pandas DataFrame.")

    if asset_returns.empty or asset_returns.shape[1] == 0:
        raise ValueError("asset_returns must not be empty.")

    if target_weights.empty:
        raise ValueError("target_weights must not be empty.")

    if not asset_returns.columns.is_unique:
        raise ValueError("Asset return columns must be unique.")

    if not asset_returns.index.is_unique:
        raise ValueError("Asset return dates must be unique.")

    if not asset_returns.index.is_monotonic_increasing:
        raise ValueError("Asset return dates must be sorted.")

    if not target_weights.columns.is_unique:
        raise ValueError("Target weight columns must be unique.")

    if not target_weights.index.is_unique:
        raise ValueError("Rebalance dates must be unique.")

    if not target_weights.index.is_monotonic_increasing:
        raise ValueError("Rebalance dates must be sorted.")

    if not asset_returns.columns.equals(target_weights.columns):
        raise ValueError(
            "Return and target weight assets must match in the same order."
        )

    if target_weights.index[0] != asset_returns.index[0]:
        raise ValueError(
            "The first rebalance date must match the first return date."
        )

    if not target_weights.index.isin(asset_returns.index).all():
        raise ValueError("All rebalance dates must exist in asset returns.")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in asset_returns.dtypes
    ):
        raise TypeError("Asset returns must be numeric.")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in target_weights.dtypes
    ):
        raise TypeError("Target weights must be numeric.")

    returns = asset_returns.to_numpy(dtype=float)
    targets = target_weights.to_numpy(dtype=float)

    if not np.isfinite(returns).all():
        raise ValueError("Asset returns must be finite.")

    if (returns <= -1.0).any():
        raise ValueError("Asset returns must be greater than -100%.")

    if not np.isfinite(targets).all():
        raise ValueError("Target weights must be finite.")

    if (targets < 0.0).any():
        raise ValueError("Target weights must be non-negative.")

    if not np.allclose(targets.sum(axis=1), 1.0):
        raise ValueError("Target weights must sum to 1.")

    daily_weights = pd.DataFrame(
        index=asset_returns.index,
        columns=asset_returns.columns,
        dtype=float,
    )

    rebalance_positions = [
        asset_returns.index.get_loc(date)
        for date in target_weights.index
    ]

    for period, start in enumerate(rebalance_positions):
        end = (
            rebalance_positions[period + 1]
            if period + 1 < len(rebalance_positions)
            else len(asset_returns)
        )

        current_weights = targets[period].copy()

        for position in range(start, end):
            date = asset_returns.index[position]

            daily_weights.loc[date] = current_weights

            daily_asset_returns = returns[position]

            gross_return = np.dot(
                current_weights,
                daily_asset_returns,
            )

            ending_values = current_weights * (
                1.0 + daily_asset_returns
            )

            current_weights = ending_values / (1.0 + gross_return)

    return daily_weights


def run_rolling_attribution(
    prices: pd.DataFrame,
    estimation_window: int = 252,
    rebalance_frequency: int = 21,
    max_weight: float = 0.40,
    transaction_cost_bps: float = 10.0,
    covariance_method: str = "sample",
) -> dict[str, pd.DataFrame]:
    """
    Run daily performance attribution for both rolling strategies.

    The existing rolling backtest remains the source of truth for
    portfolio returns and executed turnover.
    """

    backtest = run_rolling_backtest(
        prices,
        estimation_window=estimation_window,
        rebalance_frequency=rebalance_frequency,
        max_weight=max_weight,
        transaction_cost_bps=transaction_cost_bps,
        covariance_method=covariance_method,
    )

    asset_returns = calculate_returns(prices).loc[backtest.index]

    minimum_variance_targets = collect_rolling_weights(
        prices,
        estimation_window=estimation_window,
        rebalance_frequency=rebalance_frequency,
        max_weight=max_weight,
        covariance_method=covariance_method,
    )

    equal_weights = pd.Series(
        1.0 / prices.shape[1],
        index=prices.columns,
    )

    equal_targets = pd.DataFrame(
        np.tile(
            equal_weights.to_numpy(),
            (len(minimum_variance_targets), 1),
        ),
        index=minimum_variance_targets.index,
        columns=prices.columns,
    )

    target_weights = {
        "Equal Weight": equal_targets,
        "Minimum Variance": minimum_variance_targets,
    }

    results = {}

    for strategy, targets in target_weights.items():
        daily_weights = _build_daily_weights(
            asset_returns,
            targets,
        )

        gross_returns = (
            asset_returns * daily_weights
        ).sum(axis=1)

        turnover_column = f"{strategy} Turnover"
        turnover = backtest[turnover_column]

        transaction_cost_fraction = (
            turnover * transaction_cost_bps / 10_000.0
        )

        # Match the multiplicative transaction-cost convention
        # used by backtesting.rolling._apply_transaction_cost.
        transaction_cost_deductions = (
            (1.0 + gross_returns) * transaction_cost_fraction
        )

        attribution = calculate_daily_attribution(
            asset_returns,
            daily_weights,
            transaction_costs=transaction_cost_deductions,
        )

        results[strategy] = attribution

    return results
