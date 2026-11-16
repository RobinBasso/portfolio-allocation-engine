
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.rolling import run_rolling_backtest
from core.backtest_metrics import calculate_backtest_metrics


PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"

ESTIMATION_WINDOWS = [126, 252, 504]
REBALANCE_FREQUENCIES = [21, 63]

MAX_WEIGHT = 0.40
TRANSACTION_COST_BPS = 10.0

COVARIANCE_METHODS = {
    "Sample": "sample",
    "Fixed Shrinkage": "shrinkage",
    "Ledoit-Wolf": "ledoit_wolf",
}


def run_configuration(
    prices: pd.DataFrame,
    estimation_window: int,
    rebalance_frequency: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run all covariance estimators on identical evaluation dates."""

    returns = {}
    turnover = {}
    reference_dates = None

    for name, method in COVARIANCE_METHODS.items():
        backtest = run_rolling_backtest(
            prices,
            estimation_window=estimation_window,
            rebalance_frequency=rebalance_frequency,
            max_weight=MAX_WEIGHT,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            covariance_method=method,
        )

        if reference_dates is None:
            reference_dates = backtest.index
        elif not backtest.index.equals(reference_dates):
            raise ValueError(
                "Covariance estimators produced different evaluation dates."
            )

        returns[name] = backtest["Minimum Variance Net"]
        turnover[name] = backtest["Minimum Variance Turnover"].sum()

    return pd.DataFrame(returns), turnover


def main() -> None:
    """Evaluate covariance estimator robustness across configurations."""

    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Price data not found: {PRICES_PATH}")

    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    if prices.empty:
        raise ValueError("Price data is empty.")

    overall_results = []
    yearly_results = []

    reference_dates = None

    for estimation_window in ESTIMATION_WINDOWS:
        for rebalance_frequency in REBALANCE_FREQUENCIES:
            print(
                f"\nRunning window={estimation_window}, "
                f"rebalance={rebalance_frequency}..."
            )

            common_start = max(ESTIMATION_WINDOWS)

            if len(prices) <= common_start:
                raise ValueError(
                    "Insufficient prices for the common evaluation period."
                )

            configuration_prices = prices.iloc[
                common_start - estimation_window:
            ]

            returns, turnover = run_configuration(
                configuration_prices,
                estimation_window,
                rebalance_frequency,
            )

            if reference_dates is None:
                reference_dates = returns.index
            elif not returns.index.equals(reference_dates):
                raise ValueError(
                    "Configurations produced different evaluation dates."
                )

            metrics = calculate_backtest_metrics(returns)

            for estimator in COVARIANCE_METHODS:
                overall_results.append(
                    {
                        "Estimation Window": estimation_window,
                        "Rebalance Frequency": rebalance_frequency,
                        "Estimator": estimator,
                        "Annual Return": metrics.loc[
                            estimator, "Annualized Return"
                        ],
                        "Annual Volatility": metrics.loc[
                            estimator, "Annualized Volatility"
                        ],
                        "Sharpe": metrics.loc[estimator, "Sharpe Ratio"],
                        "Max Drawdown": metrics.loc[
                            estimator, "Maximum Drawdown"
                        ],
                        "Cumulative Turnover": turnover[estimator],
                    }
                )

            for year, yearly_returns in returns.groupby(returns.index.year):
                if len(yearly_returns) < 2:
                    continue

                yearly_metrics = calculate_backtest_metrics(yearly_returns)

                for estimator in COVARIANCE_METHODS:
                    yearly_results.append(
                        {
                            "Estimation Window": estimation_window,
                            "Rebalance Frequency": rebalance_frequency,
                            "Year": year,
                            "Estimator": estimator,
                            "Net Sharpe": yearly_metrics.loc[
                                estimator, "Sharpe Ratio"
                            ],
                        }
                    )

    overall = pd.DataFrame(overall_results)
    yearly = pd.DataFrame(yearly_results)

    percentage_columns = [
        "Annual Return",
        "Annual Volatility",
        "Max Drawdown",
    ]
    overall[percentage_columns] *= 100

    print("\nOVERALL NET PERFORMANCE")
    print(
        overall.round(4).to_string(index=False)
    )

    print("\nYEARLY NET SHARPE RATIOS")
    print(
        yearly.pivot_table(
            index=[
                "Estimation Window",
                "Rebalance Frequency",
                "Year",
            ],
            columns="Estimator",
            values="Net Sharpe",
        ).round(4).to_string()
    )

    print("\nHIGHEST NET SHARPE BY CONFIGURATION")

    winners = overall.loc[
        overall.groupby(
            ["Estimation Window", "Rebalance Frequency"]
        )["Sharpe"].idxmax()
    ]

    print(
        winners[
            [
                "Estimation Window",
                "Rebalance Frequency",
                "Estimator",
                "Sharpe",
            ]
        ].round(4).to_string(index=False)
    )

    print(
        "\nResults are historical and based on synthetic data. "
        "They do not represent actual investment performance."
    )


if __name__ == "__main__":
    main()
