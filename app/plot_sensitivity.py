import sys

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from core.sensitivity import run_sensitivity_analysis


PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    """Generate sensitivity heatmaps for both portfolio strategies."""

    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Price data not found: {PRICES_PATH}")

    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    results = run_sensitivity_analysis(prices)

    REPORTS_DIR.mkdir(exist_ok=True)

    for strategy in ["Equal Weight", "Minimum Variance"]:
        strategy_results = results.xs(
            strategy,
            level="Strategy",
        )

        sharpe = strategy_results["Net Sharpe Ratio"].unstack(
            "Rebalance Frequency"
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        image = ax.imshow(sharpe.to_numpy(), aspect="auto")

        ax.set_xticks(range(len(sharpe.columns)))
        ax.set_xticklabels(sharpe.columns)

        ax.set_yticks(range(len(sharpe.index)))
        ax.set_yticklabels(sharpe.index)

        ax.set_xlabel("Rebalancing Frequency (trading days)")
        ax.set_ylabel("Estimation Window (trading days)")
        ax.set_title(f"{strategy}: Net Sharpe Ratio")

        for i in range(len(sharpe.index)):
            for j in range(len(sharpe.columns)):
                ax.text(
                    j,
                    i,
                    f"{sharpe.iloc[i, j]:.3f}",
                    ha="center",
                    va="center",
                )

        fig.colorbar(image, ax=ax, label="Net Sharpe Ratio")
        fig.tight_layout()

        filename = (
            strategy.lower().replace(" ", "_")
            + "_sensitivity.png"
        )

        output_path = REPORTS_DIR / filename

        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
