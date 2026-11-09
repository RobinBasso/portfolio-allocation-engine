import sys

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from core.sensitivity import run_sensitivity_analysis


PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    """Run and save the portfolio sensitivity analysis."""

    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Price data not found: {PRICES_PATH}")

    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    if prices.empty:
        raise ValueError("Price data is empty.")

    estimation_windows = (126, 252, 504)
    rebalance_frequencies = (5, 21, 63)
    transaction_cost_bps = 10.0

    results = run_sensitivity_analysis(
        prices,
        estimation_windows=estimation_windows,
        rebalance_frequencies=rebalance_frequencies,
        transaction_cost_bps=transaction_cost_bps,
    )

    common_start = prices.index[max(estimation_windows)]
    common_end = prices.index[-1]

    REPORTS_DIR.mkdir(exist_ok=True)

    output_path = REPORTS_DIR / "sensitivity_analysis.csv"
    results.to_csv(output_path)

    print("\nPORTFOLIO SENSITIVITY ANALYSIS")
    print(f"Evaluation period: {common_start.date()} to {common_end.date()}")
    print(f"Transaction costs: {transaction_cost_bps:.1f} bps")
    print(f"Estimation windows: {estimation_windows}")
    print(f"Rebalancing frequencies: {rebalance_frequencies}")

    display_columns = [
        "Gross Annualized Return",
        "Net Annualized Return",
        "Net Annualized Volatility",
        "Net Sharpe Ratio",
        "Net Maximum Drawdown",
        "Cumulative Turnover",
    ]

    print("\nSENSITIVITY RESULTS")
    print(results[display_columns].round(4).to_string())

    print("\nBEST NET SHARPE RATIO")
    best_sharpe = results["Net Sharpe Ratio"].idxmax()
    print(f"Configuration: {best_sharpe}")
    print(f"Net Sharpe Ratio: {results.loc[best_sharpe, 'Net Sharpe Ratio']:.4f}")

    print("\nBEST NET ANNUALIZED RETURN")
    best_return = results["Net Annualized Return"].idxmax()
    print(f"Configuration: {best_return}")
    print(
        f"Net Annualized Return: "
        f"{results.loc[best_return, 'Net Annualized Return']:.4%}"
    )

    print(f"\nFull results saved to: {output_path}")

    print(
        "\nNote: Results are based on synthetic data and do not "
        "represent actual investment performance."
    )


if __name__ == "__main__":
    main()
