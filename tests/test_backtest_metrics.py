import numpy as np
import pandas as pd
import pytest

from core.backtest_metrics import calculate_backtest_metrics


@pytest.fixture
def portfolio_returns():
    return pd.DataFrame({
        "Strategy A": [0.01, -0.02, 0.015, 0.005, 0.01],
        "Strategy B": [0.005, -0.01, 0.01, 0.002, 0.008],
    })


def test_returns_metrics_for_all_portfolios(portfolio_returns):
    result = calculate_backtest_metrics(portfolio_returns)

    assert list(result.index) == ["Strategy A", "Strategy B"]
    assert list(result.columns) == [
        "Total Return",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown",
    ]


def test_total_return_is_compounded(portfolio_returns):
    result = calculate_backtest_metrics(portfolio_returns)

    expected = (1 + portfolio_returns["Strategy A"]).prod() - 1

    assert result.loc["Strategy A", "Total Return"] == pytest.approx(expected)


def test_annualized_volatility_is_calculated_correctly(portfolio_returns):
    result = calculate_backtest_metrics(portfolio_returns)

    expected = (
        portfolio_returns["Strategy A"].std(ddof=1) * np.sqrt(252)
    )

    assert result.loc[
        "Strategy A", "Annualized Volatility"
    ] == pytest.approx(expected)


def test_risk_free_rate_changes_sharpe_ratio(portfolio_returns):
    zero_rate = calculate_backtest_metrics(portfolio_returns, 0.0)
    positive_rate = calculate_backtest_metrics(portfolio_returns, 0.02)

    assert (
        positive_rate.loc["Strategy A", "Sharpe Ratio"]
        < zero_rate.loc["Strategy A", "Sharpe Ratio"]
    )


def test_maximum_drawdown_is_non_positive(portfolio_returns):
    result = calculate_backtest_metrics(portfolio_returns)

    assert (result["Maximum Drawdown"] <= 0).all()


def test_rejects_empty_returns():
    with pytest.raises(ValueError, match="empty"):
        calculate_backtest_metrics(pd.DataFrame())


def test_rejects_non_finite_returns():
    returns = pd.DataFrame({"Strategy A": [0.01, np.nan, 0.02]})

    with pytest.raises(ValueError, match="finite"):
        calculate_backtest_metrics(returns)


def test_rejects_returns_below_minus_100_percent():
    returns = pd.DataFrame({"Strategy A": [0.01, -1.1, 0.02]})

    with pytest.raises(ValueError, match="below"):
        calculate_backtest_metrics(returns)


def test_rejects_single_observation():
    returns = pd.DataFrame({"Strategy A": [0.01]})

    with pytest.raises(ValueError, match="two"):
        calculate_backtest_metrics(returns)
