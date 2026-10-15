
import numpy as np
import pandas as pd

from core.returns import calculate_returns


def historical_stress_report(
    prices: pd.DataFrame,
    portfolios: dict[str, pd.Series],
    window: int = 21,
) -> pd.DataFrame:
    """Calculate historical drawdown and adverse-period returns."""

    if prices.empty:
        raise ValueError("Price data cannot be empty.")
    if not portfolios:
        raise ValueError("At least one portfolio is required.")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("Window must be a positive integer.")

    returns = calculate_returns(prices)

    if window > len(returns):
        raise ValueError("Window cannot exceed the number of return observations.")

    results = {}

    for name, weights in portfolios.items():
        if not isinstance(weights, pd.Series):
            raise TypeError("Portfolio weights must be pandas Series.")
        if not weights.index.is_unique:
            raise ValueError("Weight asset labels must be unique.")
        if set(weights.index) != set(returns.columns):
            raise ValueError("Weight labels must match price columns.")

        weights = weights.reindex(returns.columns).astype(float)
        weight_values = weights.to_numpy()

        if not np.isfinite(weight_values).all():
            raise ValueError("Weights must be finite.")
        if np.any(weight_values < 0):
            raise ValueError("Weights must be non-negative.")
        if not np.isclose(weight_values.sum(), 1.0):
            raise ValueError("Weights must sum to one.")

        portfolio_returns = returns @ weight_values

        # Include initial wealth of 1.0 when measuring drawdown.
        wealth = np.concatenate(
            [[1.0], (1.0 + portfolio_returns).cumprod().to_numpy()]
        )
        running_peak = np.maximum.accumulate(wealth)
        drawdowns = wealth / running_peak - 1.0

        rolling_returns = (
            (1.0 + portfolio_returns)
            .rolling(window)
            .apply(np.prod, raw=True)
            - 1.0
        )

        results[name] = {
            "Maximum Drawdown": float(drawdowns.min()),
            f"Worst {window}-Day Return": float(rolling_returns.min()),
            "Worst Daily Return": float(portfolio_returns.min()),
        }

    return pd.DataFrame(results).T
