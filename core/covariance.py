import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def calculate_covariance(
    returns: pd.DataFrame,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Calculate the covariance matrix of asset returns.

    Parameters
    ----------
    returns:
        DataFrame containing periodic asset returns.
    annualize:
        If True, scale covariance to annualized values.

    Returns
    -------
    DataFrame
        Asset covariance matrix.
    """
    if returns.empty:
        raise ValueError("Return data cannot be empty.")

    covariance = returns.cov()

    if annualize:
        covariance = covariance * TRADING_DAYS_PER_YEAR

    return covariance