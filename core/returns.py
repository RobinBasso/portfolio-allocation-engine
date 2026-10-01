import numpy as np
import pandas as pd


def calculate_returns(
    prices: pd.DataFrame,
    method: str = "simple",
) -> pd.DataFrame:
    """
    Convert price data into periodic returns.

    Parameters
    ----------
    prices:
        DataFrame containing asset prices.
    method:
        "simple" for percentage returns or
        "log" for continuously compounded returns.

    Returns
    -------
    DataFrame
        Asset returns with the initial missing observation removed.
    """
    if prices.empty:
        raise ValueError("Price data cannot be empty.")

    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")

    if method == "simple":
        returns = prices.pct_change()

    elif method == "log":
        returns = np.log(prices / prices.shift(1))

    else:
        raise ValueError(
            f"Unknown return method: {method}. "
            "Use 'simple' or 'log'."
        )

    return returns.dropna()