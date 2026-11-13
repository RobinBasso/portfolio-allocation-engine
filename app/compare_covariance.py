import sys
from pathlib import Path

import pandas as pd
from sklearn.covariance import LedoitWolf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.covariance import (
    calculate_covariance,
    calculate_shrinkage_covariance,
    calculate_ledoit_wolf_covariance,
)
from core.portfolio import (
    calculate_returns,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
)
from core.risk import portfolio_volatility
from optimization.minimum_variance import minimum_variance_weights


PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"


def main() -> None:
    """Compare minimum-variance portfolios across covariance estimators."""

    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Price data not found: {PRICES_PATH}")

    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    if prices.empty:
        raise ValueError("Price data is empty.")

    returns = calculate_returns(prices)

    estimators = {
        "Sample": calculate_covariance(returns),
        "Fixed Shrinkage (20%)": calculate_shrinkage_covariance(
            returns,
            shrinkage=0.20,
        ),
        "Ledoit-Wolf": calculate_ledoit_wolf_covariance(returns),
    }

    results = {}
    allocations = {}

    for name, covariance in estimators.items():
        weights = minimum_variance_weights(covariance)
        weight_values = weights.to_numpy()

        portfolio_returns = returns @ weight_values

        results[name] = {
            "Estimated Volatility": portfolio_volatility(
                covariance,
                weight_values,
            ),
            "Realized Return": annualized_return(portfolio_returns),
            "Realized Volatility": annualized_volatility(portfolio_returns),
        }

        results[name]["Realized Sharpe"] = sharpe_ratio(
            results[name]["Realized Return"],
            results[name]["Realized Volatility"],
        )

        allocations[name] = weights

    metrics = pd.DataFrame(results).T

    print("\nCOVARIANCE ESTIMATOR COMPARISON")
    display_metrics = metrics.copy()

    percentage_columns = [
        "Estimated Volatility",
        "Realized Return",
        "Realized Volatility",
    ]

    display_metrics[percentage_columns] = (
        display_metrics[percentage_columns] * 100
    )

    print(display_metrics.round(4).to_string())

    print("\nMINIMUM-VARIANCE ALLOCATIONS")
    allocation_table = pd.DataFrame(allocations).mul(100)
    print(
        allocation_table.round(2)
        .astype(str)
        .add("%")
        .to_string()
    )

    ledoit_wolf = LedoitWolf().fit(returns.to_numpy())

    print("\nLEDOIT-WOLF SHRINKAGE INTENSITY")
    print(f"{ledoit_wolf.shrinkage_:.4f}")

    print("\nINTERPRETATION")
    print(
        "All results are in-sample. Estimated volatility comes from "
        "each estimator's covariance matrix. Realized performance "
        "uses the same historical return period for all portfolios. "
        "These results do not establish which estimator will perform "
        "best out of sample."
    )


if __name__ == "__main__":
    main()
