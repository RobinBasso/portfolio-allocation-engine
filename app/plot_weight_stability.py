
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.weight_stability import (
    COVARIANCE_METHODS,
    calculate_allocation_turnover,
    run_weight_stability_analysis,
)


PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports"


def plot_allocation_history(
    weight_histories: dict[str, pd.DataFrame],
) -> None:
    """Plot rolling asset allocations for each covariance estimator."""

    fig, axes = plt.subplots(
        nrows=len(COVARIANCE_METHODS),
        ncols=1,
        figsize=(12, 11),
        sharex=True,
        sharey=True,
    )

    for ax, method in zip(axes, COVARIANCE_METHODS):
        weights = weight_histories[method].mul(100)

        weights.plot(
            ax=ax,
            linewidth=1.5,
        )

        ax.set_title(f"{method.replace('_', ' ').title()} Covariance")
        ax.set_ylabel("Allocation (%)")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(title="Asset", loc="upper left", ncol=3)

    axes[-1].set_xlabel("Rebalance Date")

    fig.suptitle(
        "Minimum-Variance Allocation Stability",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "weight_allocation_history.png",
        dpi=160,
    )
    plt.close(fig)


def plot_allocation_variability(
    allocation_statistics: dict[str, pd.DataFrame],
) -> None:
    """Compare allocation variability across covariance estimators."""

    standard_deviations = pd.DataFrame({
        method: allocation_statistics[method]["Std"]
        for method in COVARIANCE_METHODS
    })

    ax = standard_deviations.mul(100).plot(
        kind="bar",
        figsize=(11, 6),
        width=0.8,
    )

    ax.set_title("Allocation Variability Across Covariance Estimators")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Weight Standard Deviation (%)")
    ax.legend(title="Covariance Estimator")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "weight_allocation_variability.png",
        dpi=160,
    )
    plt.close()


def plot_allocation_turnover(
    weight_histories: dict[str, pd.DataFrame],
) -> None:
    """Plot target allocation turnover between consecutive rebalances."""

    turnover_history = pd.DataFrame({
        method: calculate_allocation_turnover(
            weight_histories[method]
        )
        for method in COVARIANCE_METHODS
    })

    ax = turnover_history.mul(100).plot(
        figsize=(11, 6),
        linewidth=1.8,
        marker="o",
        markersize=3,
    )

    ax.set_title("Target Allocation Turnover")
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("One-Way Turnover (%)")
    ax.legend(title="Covariance Estimator")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "weight_allocation_turnover.png",
        dpi=160,
    )
    plt.close()


def main() -> None:
    """Generate allocation stability charts."""

    if not PRICES_PATH.exists():
        raise FileNotFoundError(
            f"Price data not found: {PRICES_PATH}"
        )

    prices = pd.read_csv(
        PRICES_PATH,
        index_col=0,
        parse_dates=True,
    )

    if prices.empty:
        raise ValueError("Price data is empty.")

    results = run_weight_stability_analysis(prices)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_allocation_history(results["weight_histories"])

    plot_allocation_variability(
        results["allocation_statistics"]
    )

    plot_allocation_turnover(results["weight_histories"])

    print(f"Weight stability charts saved to: {OUTPUT_DIR}")
    print("- weight_allocation_history.png")
    print("- weight_allocation_variability.png")
    print("- weight_allocation_turnover.png")


if __name__ == "__main__":
    main()
