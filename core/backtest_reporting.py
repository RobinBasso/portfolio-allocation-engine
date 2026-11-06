import pandas as pd

from core.backtest_metrics import calculate_backtest_metrics


def build_backtest_report(
    portfolio_returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict[str, pd.DataFrame | dict[str, str]]:
    """Build a comparative report for out-of-sample portfolio returns."""

    metrics = calculate_backtest_metrics(
        portfolio_returns,
        risk_free_rate=risk_free_rate,
    )

    if len(metrics) != 2:
        raise ValueError("Exactly two strategies are required for comparison.")

    first, second = metrics.index

    higher_is_better = [
        "Total Return",
        "Annualized Return",
        "Sharpe Ratio",
        "Maximum Drawdown",
    ]
    lower_is_better = [
        "Annualized Volatility",
    ]

    winners = {}

    for metric in higher_is_better:
        winners[metric] = (
            first if metrics.loc[first, metric] > metrics.loc[second, metric]
            else second
        )

    for metric in lower_is_better:
        winners[metric] = (
            first if metrics.loc[first, metric] < metrics.loc[second, metric]
            else second
        )

    differences = metrics.loc[first] - metrics.loc[second]
    differences.name = f"{first} - {second}"

    comparison = pd.DataFrame(
        {
            "Difference": differences,
            "Better Strategy": pd.Series(winners),
        }
    )

    interpretation = {
        "Return": winners["Annualized Return"],
        "Volatility": winners["Annualized Volatility"],
        "Sharpe Ratio": winners["Sharpe Ratio"],
        "Maximum Drawdown": winners["Maximum Drawdown"],
    }

    return {
        "metrics": metrics,
        "comparison": comparison,
        "interpretation": interpretation,
    }
