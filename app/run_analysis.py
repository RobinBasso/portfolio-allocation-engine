import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.analysis import run_portfolio_analysis

PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"


def main() -> None:
    """Load market data and print the portfolio analysis report."""
    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Price data not found: {PRICES_PATH}")

    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    if prices.empty:
        raise ValueError("Price data is empty.")

    results = run_portfolio_analysis(prices)

    print("\nPORTFOLIO PERFORMANCE")
    print((results["comparison"] * 1).round(4).to_string())

    print("\nPORTFOLIO ALLOCATIONS")
    allocations = pd.DataFrame(results["allocations"]).mul(100)
    print(allocations.round(2).astype(str).add("%").to_string())

    print("\nRISK ATTRIBUTION — MINIMUM VARIANCE")
    risk_report = results["risk_reports"]["Minimum Variance"]["asset_report"]
    print(
        risk_report.mul(100)
        .round(2)
        .astype(str)
        .add("%")
        .to_string()
    )

    print("\nHISTORICAL STRESS (%)")
    print(
        results["historical_stress"]
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
        .to_string()
    )

    print("\nPARAMETRIC STRESS (%)")
    print(
        results["parametric_stress"]
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
        .to_string()
    )


if __name__ == "__main__":
    main()
