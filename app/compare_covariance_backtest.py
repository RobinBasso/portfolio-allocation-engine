
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.rolling import run_rolling_backtest
from core.backtest_metrics import calculate_backtest_metrics


PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"

ESTIMATION_WINDOW = 252
REBALANCE_FREQUENCY = 21
MAX_WEIGHT = 0.40
TRANSACTION_COST_BPS = 10.0

COVARIANCE_METHODS = {
    "Sample": "sample",
    "Fixed Shrinkage (20%)": "shrinkage",
    "Ledoit-Wolf": "ledoit_wolf",
}


def main() -> None:
    """Compare covariance estimators using out-of-sample backtests."""

    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Price data not found: {PRICES_PATH}")

    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    if prices.empty:
        raise ValueError("Price data is empty.")

    portfolio_returns = {}
    cumulative_turnover = {}

    for name, method in COVARIANCE_METHODS.items():
        backtest = run_rolling_backtest(
            prices,
            estimation_window=ESTIMATION_WINDOW,
            rebalance_frequency=REBALANCE_FREQUENCY,
            max_weight=MAX_WEIGHT,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            covariance_method=method,
        )

        if not portfolio_returns:
            reference_dates = backtest.index
        elif not backtest.index.equals(reference_dates):
            raise ValueError(
                "Covariance estimators produced different evaluation dates."
            )

        portfolio_returns[f"{name} Gross"] = backtest["Minimum Variance"]
        portfolio_returns[f"{name} Net"] = backtest["Minimum Variance Net"]

        cumulative_turnover[name] = backtest[
            "Minimum Variance Turnover"
        ].sum()

    comparison_returns = pd.DataFrame(portfolio_returns)

    metrics = calculate_backtest_metrics(comparison_returns)

    print("\nOUT-OF-SAMPLE COVARIANCE ESTIMATOR COMPARISON")

    print(
        f"Period: {comparison_returns.index.min().date()} "
        f"to {comparison_returns.index.max().date()}"
    )
    print(f"Observations: {len(comparison_returns)}")
    print(f"Estimation window: {ESTIMATION_WINDOW} price observations")
    print(f"Rebalancing frequency: {REBALANCE_FREQUENCY} trading days")
    print(f"Maximum asset weight: {MAX_WEIGHT:.0%}")
    print(f"Transaction costs: {TRANSACTION_COST_BPS:.1f} bps")

    print("\nGROSS AND NET PERFORMANCE")

    display_metrics = metrics.copy()

    percentage_columns = [
        "Total Return",
        "Annualized Return",
        "Annualized Volatility",
        "Maximum Drawdown",
    ]

    display_metrics[percentage_columns] *= 100

    print(display_metrics.round(4).to_string())

    print("\nCUMULATIVE TURNOVER")

    turnover_table = pd.Series(
        cumulative_turnover,
        name="Cumulative Turnover",
    )

    print(turnover_table.round(4).to_string())

    print("\nINTERPRETATION")

    net_sharpe = metrics.loc[
        [f"{name} Net" for name in COVARIANCE_METHODS],
        "Sharpe Ratio",
    ]

    best_method = net_sharpe.idxmax()

    print(f"Highest net Sharpe ratio: {best_method}")
    print(
        "This is a historical result on the evaluation period, "
        "not evidence of future outperformance."
    )

    print(
        "\nNote: Results are based on synthetic data and do not "
        "represent actual investment performance."
    )


if __name__ == "__main__":
    main()
