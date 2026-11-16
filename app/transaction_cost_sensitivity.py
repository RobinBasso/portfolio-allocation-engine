
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
COMMON_START = 504

MAX_WEIGHT = 0.40

TRANSACTION_COSTS_BPS = [0.0, 10.0, 25.0, 50.0]

COVARIANCE_METHODS = {
    "Sample": "sample",
    "Fixed Shrinkage": "shrinkage",
    "Ledoit-Wolf": "ledoit_wolf",
}


def main() -> None:
    """Evaluate portfolio performance under different transaction costs."""

    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Price data not found: {PRICES_PATH}")

    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    if prices.empty or len(prices) <= COMMON_START:
        raise ValueError("Insufficient price data.")

    configuration_prices = prices.iloc[
        COMMON_START - ESTIMATION_WINDOW:
    ]

    results = []
    reference_dates = None
    baseline_backtests = {}

    for cost_bps in TRANSACTION_COSTS_BPS:
        print(f"\nRunning transaction cost={cost_bps:.0f} bps...")

        for estimator_name, covariance_method in COVARIANCE_METHODS.items():
            backtest = run_rolling_backtest(
                configuration_prices,
                estimation_window=ESTIMATION_WINDOW,
                rebalance_frequency=REBALANCE_FREQUENCY,
                max_weight=MAX_WEIGHT,
                transaction_cost_bps=cost_bps,
                covariance_method=covariance_method,
            )

            if reference_dates is None:
                reference_dates = backtest.index
            elif not backtest.index.equals(reference_dates):
                raise ValueError(
                    "Configurations produced different evaluation dates."
                )

            if cost_bps == 0.0:
                baseline_backtests[estimator_name] = backtest.copy()

                pd.testing.assert_series_equal(
                    backtest["Minimum Variance"],
                    backtest["Minimum Variance Net"],
                    check_names=False,
                )
            else:
                baseline = baseline_backtests[estimator_name]

                pd.testing.assert_series_equal(
                    baseline["Minimum Variance"],
                    backtest["Minimum Variance"],
                )

                pd.testing.assert_series_equal(
                    baseline["Minimum Variance Turnover"],
                    backtest["Minimum Variance Turnover"],
                )

            net_returns = backtest["Minimum Variance Net"]

            metrics = calculate_backtest_metrics(
                net_returns.to_frame(estimator_name)
            )

            results.append(
                {
                    "Transaction Cost (bps)": cost_bps,
                    "Estimator": estimator_name,
                    "Annual Return": metrics.loc[
                        estimator_name, "Annualized Return"
                    ],
                    "Annual Volatility": metrics.loc[
                        estimator_name, "Annualized Volatility"
                    ],
                    "Sharpe": metrics.loc[
                        estimator_name, "Sharpe Ratio"
                    ],
                    "Max Drawdown": metrics.loc[
                        estimator_name, "Maximum Drawdown"
                    ],
                    "Cumulative Turnover": backtest[
                        "Minimum Variance Turnover"
                    ].sum(),
                }
            )

    results_df = pd.DataFrame(results)

    percentage_columns = [
        "Annual Return",
        "Annual Volatility",
        "Max Drawdown",
    ]
    results_df[percentage_columns] *= 100

    print("\nTRANSACTION COST SENSITIVITY")
    print(results_df.round(4).to_string(index=False))

    print("\nValidation passed:")
    print("- Evaluation dates are identical across configurations.")
    print("- Gross returns are invariant to transaction costs.")
    print("- Portfolio turnover is invariant to transaction costs.")
    print("- Net returns equal gross returns at zero transaction costs.")

    print(
        "\nResults are historical and based on synthetic data. "
        "They do not represent actual investment performance."
    )


if __name__ == "__main__":
    main()
