
import numpy as np
import pandas as pd

from backtesting.weights import collect_rolling_weights


ESTIMATION_WINDOW = 252
REBALANCE_FREQUENCY = 21
COMMON_OOS_START = 504
MAX_WEIGHT = 0.40

COVARIANCE_METHODS = [
    "sample",
    "shrinkage",
    "ledoit_wolf",
]


def calculate_weight_statistics(
    weight_history: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate allocation statistics for each asset."""

    statistics = pd.DataFrame({
        "Mean": weight_history.mean(),
        "Std": weight_history.std(),
        "Minimum": weight_history.min(),
        "Maximum": weight_history.max(),
        "Range": weight_history.max() - weight_history.min(),
    })

    return statistics


def calculate_allocation_turnover(
    weight_history: pd.DataFrame,
) -> pd.Series:
    """Calculate one-way turnover between consecutive target allocations."""

    changes = weight_history.diff().abs().sum(axis=1) / 2

    return changes.iloc[1:]


def calculate_effective_number_of_assets(
    weight_history: pd.DataFrame,
) -> pd.Series:
    """Calculate effective number of assets from the Herfindahl index."""

    herfindahl_index = weight_history.pow(2).sum(axis=1)

    return 1 / herfindahl_index


def run_weight_stability_analysis(
    prices: pd.DataFrame,
) -> dict[str, object]:
    """Compare portfolio allocation stability across covariance estimators."""

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    if len(prices) <= COMMON_OOS_START:
        raise ValueError("Insufficient observations for common evaluation period")

    configuration_prices = prices.iloc[
        COMMON_OOS_START - ESTIMATION_WINDOW:
    ]

    weight_histories = {}
    allocation_statistics = {}
    turnover_statistics = {}
    diversification_statistics = {}

    for method in COVARIANCE_METHODS:
        weights = collect_rolling_weights(
            configuration_prices,
            estimation_window=ESTIMATION_WINDOW,
            rebalance_frequency=REBALANCE_FREQUENCY,
            max_weight=MAX_WEIGHT,
            covariance_method=method,
        )

        weight_histories[method] = weights

        allocation_statistics[method] = calculate_weight_statistics(
            weights
        )

        turnover = calculate_allocation_turnover(weights)

        turnover_statistics[method] = {
            "Mean": turnover.mean(),
            "Median": turnover.median(),
            "Maximum": turnover.max(),
        }

        effective_assets = calculate_effective_number_of_assets(
            weights
        )

        diversification_statistics[method] = {
            "Mean": effective_assets.mean(),
            "Minimum": effective_assets.min(),
            "Maximum": effective_assets.max(),
        }

    return {
        "weight_histories": weight_histories,
        "allocation_statistics": allocation_statistics,
        "turnover_statistics": pd.DataFrame(turnover_statistics).T,
        "diversification_statistics": pd.DataFrame(
            diversification_statistics
        ).T,
    }


def main():
    prices = pd.read_csv(
        "data/prices.csv",
        index_col=0,
        parse_dates=True,
    )

    results = run_weight_stability_analysis(prices)

    print("\nWEIGHT STABILITY ANALYSIS")
    print("=" * 70)

    for method, statistics in results["allocation_statistics"].items():
        print(f"\n{method.upper()} — ASSET ALLOCATION")
        print(statistics.round(4).to_string())

    print("\nALLOCATION TURNOVER")
    print(results["turnover_statistics"].round(4).to_string())

    print("\nEFFECTIVE NUMBER OF ASSETS")
    print(results["diversification_statistics"].round(4).to_string())

    print("\nWEIGHT HISTORY")
    for method, weights in results["weight_histories"].items():
        print(
            f"{method}: {len(weights)} rebalances, "
            f"{weights.index[0].date()} to {weights.index[-1].date()}"
        )


if __name__ == "__main__":
    main()
