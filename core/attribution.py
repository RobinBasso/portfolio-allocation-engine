
import numpy as np
import pandas as pd


def calculate_buy_and_hold_attribution(
    asset_returns: pd.DataFrame,
    initial_weights: pd.Series,
) -> pd.DataFrame:
    """
    Calculate exact cumulative return attribution for a buy-and-hold portfolio.

    Each asset's contribution is its initial portfolio weight multiplied
    by its cumulative asset return.

    The sum of asset contributions equals the portfolio's cumulative return.
    """

    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns must be a pandas DataFrame.")

    if not isinstance(initial_weights, pd.Series):
        raise TypeError("initial_weights must be a pandas Series.")

    if asset_returns.empty:
        raise ValueError("asset_returns must not be empty.")

    if asset_returns.shape[1] == 0:
        raise ValueError("asset_returns must contain at least one asset.")

    if not asset_returns.columns.is_unique:
        raise ValueError("Asset columns must be unique.")

    if not asset_returns.index.is_unique:
        raise ValueError("Return observations must have unique indices.")

    if not initial_weights.index.is_unique:
        raise ValueError("Weight indices must be unique.")

    if not asset_returns.columns.equals(initial_weights.index):
        raise ValueError(
            "Asset columns and weight indices must match in the same order."
        )

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in asset_returns.dtypes):
        raise TypeError("All asset returns must be numeric.")

    if not pd.api.types.is_numeric_dtype(initial_weights.dtype):
        raise TypeError("Portfolio weights must be numeric.")

    returns_array = asset_returns.to_numpy(dtype=float)
    weights_array = initial_weights.to_numpy(dtype=float)

    if not np.isfinite(returns_array).all():
        raise ValueError("Asset returns must contain only finite values.")

    if not np.isfinite(weights_array).all():
        raise ValueError("Portfolio weights must contain only finite values.")

    if (returns_array <= -1.0).any():
        raise ValueError("Asset returns must be greater than -100%.")

    if (weights_array < 0.0).any():
        raise ValueError("Portfolio weights must be non-negative.")

    if not np.isclose(weights_array.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    cumulative_asset_returns = (
        (1.0 + asset_returns).prod(axis=0) - 1.0
    )

    contributions = initial_weights * cumulative_asset_returns

    portfolio_return = contributions.sum()

    if np.isclose(portfolio_return, 0.0):
        contribution_percentages = pd.Series(
            np.nan,
            index=asset_returns.columns,
            dtype=float,
        )
    else:
        contribution_percentages = contributions / portfolio_return * 100.0

    attribution = pd.DataFrame(
        {
            "initial_weight": initial_weights,
            "cumulative_asset_return": cumulative_asset_returns,
            "return_contribution": contributions,
            "contribution_percentage": contribution_percentages,
        }
    )

    return attribution
