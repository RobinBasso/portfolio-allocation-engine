import numpy as np
import pandas as pd

from backtesting.rolling import run_rolling_backtest
from core.backtest_metrics import calculate_backtest_metrics
from core.returns import calculate_returns


ROLLING_STRATEGIES = [
    "Equal Weight",
    "Minimum Variance",
    "Equal Weight Net",
    "Minimum Variance Net",
]

BENCHMARK = "Buy and Hold Equal Weight"


def _buy_and_hold_returns(
    asset_returns: pd.DataFrame,
    initial_weights: np.ndarray,
) -> pd.Series:
    """Calculate returns for a portfolio with no intra-period rebalancing."""

    asset_growth = (1 + asset_returns).cumprod()
    portfolio_value = asset_growth @ initial_weights
    previous_value = portfolio_value.shift(1, fill_value=1.0)

    return (portfolio_value / previous_value - 1).rename(BENCHMARK)


def run_robustness_analysis(
    prices: pd.DataFrame,
    estimation_window: int = 252,
    rebalance_frequency: int = 21,
    transaction_cost_bps: float = 10.0,
    max_weight: float = 0.40,
    risk_free_rate: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Compare rolling strategies with buy-and-hold and across calendar years."""

    rolling_returns = run_rolling_backtest(
        prices,
        estimation_window=estimation_window,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=transaction_cost_bps,
        max_weight=max_weight,
    )

    asset_returns = calculate_returns(
        prices.iloc[estimation_window - 1:]
    )

    asset_returns = asset_returns.loc[rolling_returns.index]

    equal_weights = np.full(
        asset_returns.shape[1],
        1.0 / asset_returns.shape[1],
    )

    benchmark_returns = _buy_and_hold_returns(
        asset_returns,
        equal_weights,
    )

    comparison_returns = rolling_returns[
        ROLLING_STRATEGIES
    ].copy()

    comparison_returns[BENCHMARK] = benchmark_returns

    full_period_metrics = calculate_backtest_metrics(
        comparison_returns,
        risk_free_rate=risk_free_rate,
    )

    annual_results = []

    for year, yearly_returns in comparison_returns.groupby(
        comparison_returns.index.year
    ):
        if len(yearly_returns) < 2:
            continue

        yearly_metrics = calculate_backtest_metrics(
            yearly_returns,
            risk_free_rate=risk_free_rate,
        )

        yearly_metrics.index.name = "Strategy"
        yearly_metrics["Year"] = year

        annual_results.append(yearly_metrics.reset_index())

    if not annual_results:
        raise ValueError("Insufficient observations for annual analysis.")

    annual_metrics = (
        pd.concat(annual_results, ignore_index=True)
        .set_index(["Year", "Strategy"])
        .sort_index()
    )

    return {
        "returns": comparison_returns,
        "full_period_metrics": full_period_metrics,
        "annual_metrics": annual_metrics,
    }
