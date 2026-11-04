import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def calculate_backtest_metrics(
    portfolio_returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Calculate performance metrics from out-of-sample daily returns."""

    if not isinstance(portfolio_returns, pd.DataFrame) or portfolio_returns.empty:
        raise ValueError("Portfolio returns cannot be empty.")

    if len(portfolio_returns) < 2:
        raise ValueError("At least two return observations are required.")

    if not portfolio_returns.columns.is_unique:
        raise ValueError("Portfolio labels must be unique.")

    values = portfolio_returns.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Portfolio returns must be finite.")

    if np.any(values < -1):
        raise ValueError("Returns cannot be below -100%.")

    if not np.isfinite(risk_free_rate):
        raise ValueError("Risk-free rate must be finite.")

    results = {}

    for portfolio in portfolio_returns.columns:
        returns = portfolio_returns[portfolio]

        total_return = (1 + returns).prod() - 1

        annualized_return = (
            (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1
        )

        annualized_volatility = (
            returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

        if annualized_volatility > 0:
            sharpe_ratio = (
                returns.mean() * TRADING_DAYS_PER_YEAR - risk_free_rate
            ) / annualized_volatility
        else:
            sharpe_ratio = np.nan

        wealth = (1 + returns).cumprod()
        running_maximum = wealth.cummax()
        drawdown = wealth / running_maximum - 1
        maximum_drawdown = drawdown.min()

        results[portfolio] = {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Maximum Drawdown": maximum_drawdown,
        }

    return pd.DataFrame(results).T