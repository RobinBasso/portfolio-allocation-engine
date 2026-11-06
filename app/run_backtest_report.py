import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.rolling import run_rolling_backtest
from core.backtest_reporting import build_backtest_report

PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"


def main() -> None:
    """Run the rolling backtest and print its performance report."""
    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Price data not found: {PRICES_PATH}")

    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    if prices.empty:
        raise ValueError("Price data is empty.")

    portfolio_returns = run_rolling_backtest(prices)
    report = build_backtest_report(portfolio_returns)

    print("\nOUT-OF-SAMPLE BACKTEST")
    print(f"Period: {portfolio_returns.index.min().date()} "
          f"to {portfolio_returns.index.max().date()}")
    print(f"Observations: {len(portfolio_returns)}")

    print("\nPERFORMANCE METRICS")
    print(report["metrics"].round(4).to_string())

    print("\nSTRATEGY COMPARISON")
    print(report["comparison"].to_string())

    print("\nINTERPRETATION")
    for metric, strategy in report["interpretation"].items():
        print(f"{metric}: {strategy}")

    print(
        "\nNote: Results are based on synthetic data and do not "
        "represent actual investment performance."
    )


if __name__ == "__main__":
    main()
