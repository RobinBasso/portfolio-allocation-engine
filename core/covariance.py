import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def calculate_covariance(
    returns: pd.DataFrame,
    annualize: bool = True,
) -> pd.DataFrame:
    """Calculate the sample covariance matrix of asset returns."""

    if returns.empty:
        raise ValueError("Return data cannot be empty.")

    covariance = returns.cov()

    if annualize:
        covariance = covariance * TRADING_DAYS_PER_YEAR

    return covariance


def calculate_shrinkage_covariance(
    returns: pd.DataFrame,
    shrinkage: float = 0.20,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Calculate a diagonal-shrinkage covariance matrix.

    The estimator blends the sample covariance matrix with its
    diagonal target using a fixed shrinkage intensity.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Returns must be a pandas DataFrame.")

    if returns.empty or returns.shape[1] == 0:
        raise ValueError("Return data cannot be empty.")

    if not returns.columns.is_unique:
        raise ValueError("Asset labels must be unique.")

    if len(returns) < 2:
        raise ValueError("At least two observations are required.")

    if returns.isna().any().any():
        raise ValueError("Return data cannot contain missing values.")

    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("Return data must be finite.")

    if not isinstance(shrinkage, (int, float)) or isinstance(
        shrinkage, bool
    ):
        raise TypeError("Shrinkage intensity must be numeric.")

    if not np.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("Shrinkage intensity must be between 0 and 1.")

    sample_covariance = returns.cov()

    diagonal_target = pd.DataFrame(
        np.diag(np.diag(sample_covariance.to_numpy())),
        index=sample_covariance.index,
        columns=sample_covariance.columns,
    )

    covariance = (
        (1.0 - shrinkage) * sample_covariance
        + shrinkage * diagonal_target
    )

    if annualize:
        covariance = covariance * TRADING_DAYS_PER_YEAR

    return covariance
