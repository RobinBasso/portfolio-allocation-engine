
import pandas as pd

from core.comparison import compare_portfolios
from core.covariance import calculate_covariance
from core.reporting import build_allocation_report
from core.returns import calculate_returns
from optimization.minimum_variance import minimum_variance_weights
from scenarios.historical import historical_stress_report
from scenarios.parametric import parametric_stress_report


def run_portfolio_analysis(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.0,
    stress_window: int = 21,
) -> dict[str, pd.DataFrame | pd.Series | dict]:
    """Run portfolio comparison, allocation, risk attribution and stress tests."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("Prices must be a pandas DataFrame.")

    if prices.empty or prices.shape[1] == 0:
        raise ValueError("Prices must contain at least one asset.")

    if not prices.columns.is_unique:
        raise ValueError("Asset labels must be unique.")

    if not prices.index.is_unique:
        raise ValueError("Price dates must be unique.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be in ascending order.")

    if not isinstance(risk_free_rate, (int, float)) or isinstance(
        risk_free_rate, bool
    ):
        raise TypeError("Risk-free rate must be numeric.")

    if not pd.notna(risk_free_rate) or abs(risk_free_rate) == float("inf"):
        raise ValueError("Risk-free rate must be finite.")

    if not isinstance(stress_window, int) or isinstance(stress_window, bool):
        raise TypeError("Stress window must be an integer.")

    returns = calculate_returns(prices)

    if returns.empty:
        raise ValueError("Prices must contain enough observations to calculate returns.")

    covariance = calculate_covariance(returns)

    equal_weight = pd.Series(
        1.0 / len(prices.columns),
        index=prices.columns,
        name="Equal Weight",
    )

    minimum_variance = minimum_variance_weights(covariance)
    minimum_variance.name = "Minimum Variance"

    portfolios = {
        "Equal Weight": equal_weight,
        "Minimum Variance": minimum_variance,
    }

    allocation_reports = {
        name: build_allocation_report(
            prices,
            weights,
            risk_free_rate=risk_free_rate,
        )
        for name, weights in portfolios.items()
    }

    return {
        "comparison": compare_portfolios(prices),
        "allocations": portfolios,
        "risk_reports": allocation_reports,
        "historical_stress": historical_stress_report(
            prices,
            portfolios,
            window=stress_window,
        ),
        "parametric_stress": parametric_stress_report(portfolios),
    }
