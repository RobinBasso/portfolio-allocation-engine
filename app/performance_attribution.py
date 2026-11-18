
from pathlib import Path

import pandas as pd

from core.attribution import calculate_buy_and_hold_attribution
from core.covariance import calculate_covariance
from core.returns import calculate_returns
from optimization.minimum_variance import minimum_variance_weights


ESTIMATION_WINDOW = 252
MAX_WEIGHT = 0.40

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "prices.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def run_performance_attribution(
    prices: pd.DataFrame,
    estimation_window: int = ESTIMATION_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare buy-and-hold performance attribution for Equal Weight
    and Minimum Variance portfolios.

    Minimum Variance weights are estimated using the initial
    estimation window. Both portfolios are then evaluated over
    the same out-of-sample period.
    """

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")

    if prices.empty:
        raise ValueError("prices must not be empty.")

    if not prices.index.is_unique:
        raise ValueError("Price dates must be unique.")

    if not prices.columns.is_unique:
        raise ValueError("Asset columns must be unique.")

    if len(prices.columns) < 2:
        raise ValueError("At least two assets are required.")

    if estimation_window < 2:
        raise ValueError("estimation_window must be at least 2.")

    if len(prices) <= estimation_window + 1:
        raise ValueError(
            "Insufficient observations for estimation and "
            "out-of-sample attribution."
        )

    prices = prices.sort_index()

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in prices.dtypes
    ):
        raise TypeError("All price columns must be numeric.")

    if prices.isna().any().any():
        raise ValueError("Prices must not contain missing values.")

    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")

    returns = calculate_returns(prices, method="simple")

    estimation_returns = returns.iloc[:estimation_window]
    out_of_sample_returns = returns.iloc[estimation_window:]

    if out_of_sample_returns.empty:
        raise ValueError("No out-of-sample returns available.")

    covariance = calculate_covariance(
        estimation_returns,
        annualize=False,
    )

    minimum_variance = minimum_variance_weights(
        covariance,
        max_weight=MAX_WEIGHT,
    )

    equal_weight = pd.Series(
        1.0 / len(prices.columns),
        index=prices.columns,
        name="Equal Weight",
    )

    minimum_variance.name = "Minimum Variance"

    attribution_results = {}

    for portfolio_name, weights in {
        "Equal Weight": equal_weight,
        "Minimum Variance": minimum_variance,
    }.items():

        attribution = calculate_buy_and_hold_attribution(
            out_of_sample_returns,
            weights,
        )

        attribution_results[portfolio_name] = attribution

    attribution_table = pd.concat(
        attribution_results,
        names=["portfolio", "asset"],
    )

    performance_summary = pd.DataFrame(
        {
            portfolio_name: {
                "Cumulative Return": (
                    attribution["return_contribution"].sum()
                ),
                "Initial Portfolio Weight": (
                    attribution["initial_weight"].sum()
                ),
            }
            for portfolio_name, attribution
            in attribution_results.items()
        }
    ).T

    performance_summary.index.name = "Portfolio"

    return attribution_table, performance_summary


def main() -> None:
    prices = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True,
    )

    attribution, summary = run_performance_attribution(prices)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "performance_attribution.csv"
    attribution.to_csv(output_path)

    print("\nOUT-OF-SAMPLE PERFORMANCE ATTRIBUTION")
    print("=" * 65)

    print(f"Estimation window: {ESTIMATION_WINDOW} trading days")
    print(f"Maximum asset weight: {MAX_WEIGHT:.0%}")
    print(f"Attribution start: {prices.index[ESTIMATION_WINDOW + 1].date()}")
    print(f"Attribution end: {prices.index[-1].date()}")

    print("\nPORTFOLIO PERFORMANCE")
    print(summary.to_string(float_format=lambda x: f"{x:.4%}"))

    print("\nASSET-LEVEL CONTRIBUTIONS")
    print(
        attribution[
            [
                "initial_weight",
                "cumulative_asset_return",
                "return_contribution",
                "contribution_percentage",
            ]
        ].to_string(float_format=lambda x: f"{x:.4f}")
    )

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
