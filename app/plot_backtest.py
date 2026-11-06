import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.rolling import run_rolling_backtest

PRICES_PATH = PROJECT_ROOT / "data" / "prices.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports"


def plot_equity_curve(portfolio_returns: pd.DataFrame) -> None:
    """Plot cumulative portfolio performance."""
    cumulative_returns = (1 + portfolio_returns).cumprod()

    ax = cumulative_returns.plot(
        figsize=(11, 6),
        linewidth=1.8,
    )
    ax.set_title("Out-of-Sample Portfolio Performance")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(title="Strategy")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "backtest_equity_curve.png", dpi=160)
    plt.close()


def plot_drawdowns(portfolio_returns: pd.DataFrame) -> None:
    """Plot portfolio drawdowns from previous wealth peaks."""
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_peaks = cumulative_returns.cummax()
    drawdowns = cumulative_returns / running_peaks - 1

    ax = drawdowns.plot(
        figsize=(11, 6),
        linewidth=1.5,
    )
    ax.set_title("Out-of-Sample Portfolio Drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    ax.legend(title="Strategy")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "backtest_drawdowns.png", dpi=160)
    plt.close()


def main() -> None:
    """Generate rolling backtest performance charts."""
    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Price data not found: {PRICES_PATH}")

    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    if prices.empty:
        raise ValueError("Price data is empty.")

    portfolio_returns = run_rolling_backtest(prices)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_equity_curve(portfolio_returns)
    plot_drawdowns(portfolio_returns)

    print(f"Backtest charts saved to: {OUTPUT_DIR}")
    print("- backtest_equity_curve.png")
    print("- backtest_drawdowns.png")


if __name__ == "__main__":
    main()
