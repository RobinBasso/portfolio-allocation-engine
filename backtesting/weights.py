import pandas as pd

from backtesting.rolling import COVARIANCE_ESTIMATORS
from core.returns import calculate_returns
from optimization.minimum_variance import minimum_variance_weights


def collect_rolling_weights(
    prices: pd.DataFrame,
    estimation_window: int = 252,
    rebalance_frequency: int = 21,
    max_weight: float | dict[str, float] = 0.40,
    covariance_method: str = "sample",
) -> pd.DataFrame:
    """Collect minimum-variance target weights at each rebalance date."""

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    if prices.empty:
        raise ValueError("prices must not be empty")

    if prices.shape[1] < 2:
        raise ValueError("prices must contain at least two assets")

    if not prices.columns.is_unique:
        raise ValueError("asset columns must be unique")

    if not prices.index.is_unique:
        raise ValueError("price index must be unique")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("price index must be sorted in ascending order")

    if not prices.apply(lambda column: pd.api.types.is_numeric_dtype(column)).all():
        raise TypeError("all price columns must be numeric")

    if not prices.notna().all().all():
        raise ValueError("prices must not contain missing values")

    if not (prices > 0).all().all():
        raise ValueError("prices must be strictly positive")

    if estimation_window < 2:
        raise ValueError("estimation_window must be at least 2")

    if rebalance_frequency < 1:
        raise ValueError("rebalance_frequency must be at least 1")

    if len(prices) <= estimation_window:
        raise ValueError("insufficient observations for out-of-sample weights")

    if covariance_method not in COVARIANCE_ESTIMATORS:
        raise ValueError(
            f"Unknown covariance method: {covariance_method}. "
            f"Choose from {list(COVARIANCE_ESTIMATORS)}"
        )

    covariance_estimator = COVARIANCE_ESTIMATORS[covariance_method]

    records = []
    rebalance_dates = []

    train_end = estimation_window

    while train_end < len(prices):
        training_prices = prices.iloc[
            train_end - estimation_window : train_end
        ]

        training_returns = calculate_returns(training_prices)

        covariance = covariance_estimator(
            training_returns,
            annualize=False,
        )

        weights = minimum_variance_weights(
            covariance,
            max_weight=max_weight,
        )

        records.append(weights.to_numpy())
        rebalance_dates.append(prices.index[train_end])

        train_end += rebalance_frequency

    weight_history = pd.DataFrame(
        records,
        index=rebalance_dates,
        columns=prices.columns,
    )

    weight_history.index.name = "Date"

    return weight_history