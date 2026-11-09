import pandas as pd

from backtesting.rolling import run_rolling_backtest
from core.backtest_metrics import calculate_backtest_metrics


GROSS_STRATEGIES = ["Equal Weight", "Minimum Variance"]

NET_STRATEGIES = ["Equal Weight Net", "Minimum Variance Net"]

STRATEGY_PAIRS = {
    "Equal Weight": ("Equal Weight", "Equal Weight Net"),
    "Minimum Variance": ("Minimum Variance", "Minimum Variance Net"),
}


def run_sensitivity_analysis(
    prices: pd.DataFrame,
    estimation_windows: tuple[int, ...] = (126, 252, 504),
    rebalance_frequencies: tuple[int, ...] = (5, 21, 63),
    transaction_cost_bps: float = 10.0,
    max_weight: float = 0.40,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Compare backtest performance across estimation and rebalancing settings."""

    if not estimation_windows or not rebalance_frequencies:
        raise ValueError("Sensitivity parameter lists cannot be empty.")

    if len(set(estimation_windows)) != len(estimation_windows):
        raise ValueError("Estimation windows must be unique.")

    if len(set(rebalance_frequencies)) != len(rebalance_frequencies):
        raise ValueError("Rebalancing frequencies must be unique.")

    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 2
        for value in estimation_windows
    ):
        raise ValueError("Estimation windows must be integers of at least 2.")

    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in rebalance_frequencies
    ):
        raise ValueError("Rebalancing frequencies must be positive integers.")

    if len(prices) <= max(estimation_windows):
        raise ValueError("Insufficient prices for the largest estimation window.")

    common_start = prices.index[max(estimation_windows)]

    results = []

    for estimation_window in estimation_windows:
        for rebalance_frequency in rebalance_frequencies:

            backtest = run_rolling_backtest(
                prices,
                estimation_window=estimation_window,
                rebalance_frequency=rebalance_frequency,
                max_weight=max_weight,
                transaction_cost_bps=transaction_cost_bps,
            )

            # Align all configurations to the same out-of-sample period.
            backtest = backtest.loc[common_start:]

            return_columns = GROSS_STRATEGIES + NET_STRATEGIES

            metrics = calculate_backtest_metrics(
                backtest[return_columns],
                risk_free_rate=risk_free_rate,
            )

            for strategy, (gross_name, net_name) in STRATEGY_PAIRS.items():
                row = {
                    "Estimation Window": estimation_window,
                    "Rebalance Frequency": rebalance_frequency,
                    "Strategy": strategy,
                }

                for metric in metrics.columns:
                    row[f"Gross {metric}"] = metrics.loc[gross_name, metric]
                    row[f"Net {metric}"] = metrics.loc[net_name, metric]

                turnover_column = f"{strategy} Turnover"
                row["Cumulative Turnover"] = backtest[turnover_column].sum()

                results.append(row)

    return (
        pd.DataFrame(results)
        .set_index(
            ["Estimation Window", "Rebalance Frequency", "Strategy"]
        )
        .sort_index()
    )
