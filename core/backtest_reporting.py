import numpy as np
import pandas as pd

from core.backtest_metrics import calculate_backtest_metrics


def build_backtest_report(
    portfolio_returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict[str, pd.DataFrame | dict[str, str]]:
    """Build a comparative report for out-of-sample portfolio returns."""

    if not isinstance(portfolio_returns, pd.DataFrame):
        raise ValueError("Portfolio returns must be a DataFrame.")

    if portfolio_returns.empty:
        raise ValueError("Portfolio returns must not be empty.")

    if not np.isfinite(portfolio_returns.to_numpy(dtype=float)).all():
        raise ValueError("Portfolio returns must contain only finite values.")
    
    gross_strategies = ["Equal Weight", "Minimum Variance"]
    net_strategies = ["Equal Weight Net", "Minimum Variance Net"]

    if len(portfolio_returns.columns) == 2:
        strategies = gross_strategies
    elif all(
        strategy in portfolio_returns.columns
        for strategy in gross_strategies + net_strategies
    ):
        strategies = gross_strategies + net_strategies
    else:
        raise ValueError(
            "Exactly two strategies or all four gross/net return "
            "series are required."
        )

    metrics = calculate_backtest_metrics(
        portfolio_returns[strategies],
        risk_free_rate=risk_free_rate,
    )

    higher_is_better = [
        "Total Return",
        "Annualized Return",
        "Sharpe Ratio",
        "Maximum Drawdown",
    ]

    lower_is_better = ["Annualized Volatility"]

    gross_metrics = metrics.loc[gross_strategies]

    winners = {}

    for metric in higher_is_better:
        winners[metric] = gross_metrics[metric].idxmax()

    for metric in lower_is_better:
        winners[metric] = gross_metrics[metric].idxmin()

    differences = (
        gross_metrics.loc["Equal Weight"]
        - gross_metrics.loc["Minimum Variance"]
    )

    differences.name = "Equal Weight - Minimum Variance"

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

    report = {
        "metrics": metrics,
        "comparison": comparison,
        "interpretation": interpretation,
    }

    if len(strategies) == 4:
        transaction_cost_impact = pd.DataFrame(
            {
                "Annualized Return Lost": [
                    metrics.loc[gross, "Annualized Return"]
                    - metrics.loc[net, "Annualized Return"]
                    for gross, net in zip(
                        gross_strategies,
                        net_strategies,
                    )
                ],
                "Total Return Lost": [
                    metrics.loc[gross, "Total Return"]
                    - metrics.loc[net, "Total Return"]
                    for gross, net in zip(
                        gross_strategies,
                        net_strategies,
                    )
                ],
            },
            index=gross_strategies,
        )

        report["transaction_cost_impact"] = transaction_cost_impact

    return report
