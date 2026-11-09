import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.robustness import run_robustness_analysis


PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    """Run the robustness and benchmark comparison."""

    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Price data not found: {PRICES_PATH}")

    prices = pd.read_csv(
        PRICES_PATH,
        index_col=0,
        parse_dates=True,
    )

    results = run_robustness_analysis(prices)

    REPORTS_DIR.mkdir(exist_ok=True)

    results["full_period_metrics"].to_csv(
        REPORTS_DIR / "robustness_full_period.csv"
    )

    results["annual_metrics"].to_csv(
        REPORTS_DIR / "robustness_annual.csv"
    )

    print("\nROBUSTNESS ANALYSIS")
    print(
        f"Period: {results['returns'].index.min().date()} "
        f"to {results['returns'].index.max().date()}"
    )
    print(f"Observations: {len(results['returns'])}")

    print("\nFULL-PERIOD PERFORMANCE")
    print(results["full_period_metrics"].round(4).to_string())

    print("\nCALENDAR-YEAR PERFORMANCE")
    print(results["annual_metrics"].round(4).to_string())

    print("\nSaved reports:")
    print("reports/robustness_full_period.csv")
    print("reports/robustness_annual.csv")

    print(
        "\nNote: Results are based on synthetic data and do not "
        "represent actual investment performance."
    )


if __name__ == "__main__":
    main()
