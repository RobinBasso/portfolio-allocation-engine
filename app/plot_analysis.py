import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.analysis import run_portfolio_analysis

PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    """Generate portfolio allocation and risk attribution charts."""
    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Price data not found: {PRICES_PATH}")

    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    if prices.empty:
        raise ValueError("Price data is empty.")

    results = run_portfolio_analysis(prices)
    allocations = pd.DataFrame(results["allocations"])
    risk_report = results["risk_reports"]["Minimum Variance"]["asset_report"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ax = allocations.mul(100).plot(
        kind="bar",
        figsize=(10, 6),
        width=0.8,
    )
    ax.set_title("Portfolio Allocation")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Allocation (%)")
    ax.legend(title="Portfolio")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "portfolio_allocations.png", dpi=160)
    plt.close()

    risk_comparison = risk_report[["Allocation", "Risk Contribution"]].mul(100)
    ax = risk_comparison.plot(
        kind="bar",
        figsize=(10, 6),
        width=0.8,
    )
    ax.set_title("Minimum-Variance Portfolio: Allocation vs Risk Contribution")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Portfolio share (%)")
    ax.legend(title="Measure")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "risk_attribution.png", dpi=160)
    plt.close()

    print(f"Charts saved to: {OUTPUT_DIR}")
    print("- portfolio_allocations.png")
    print("- risk_attribution.png")


if __name__ == "__main__":
    main()
