import numpy as np
import pandas as pd
import pytest

from core.backtest_reporting import build_backtest_report


@pytest.fixture
def portfolio_returns():
    dates = pd.bdate_range("2024-01-01", periods=252)

    return pd.DataFrame(
        {
            "Equal Weight": np.tile([0.01, -0.005], 126),
            "Minimum Variance": np.tile([0.006, -0.002], 126),
        },
        index=dates,
    )


def test_build_backtest_report_returns_expected_sections(portfolio_returns):
    report = build_backtest_report(portfolio_returns)

    assert set(report) == {"metrics", "comparison", "interpretation"}
    assert len(report["metrics"]) == 2
    assert len(report["comparison"]) == 5


def test_report_identifies_lower_volatility_strategy(portfolio_returns):
    report = build_backtest_report(portfolio_returns)

    assert (
        report["interpretation"]["Volatility"]
        == "Minimum Variance"
    )


def test_report_identifies_higher_sharpe_strategy(portfolio_returns):
    report = build_backtest_report(portfolio_returns)

    assert (
        report["interpretation"]["Sharpe Ratio"]
        == "Minimum Variance"
    )


def test_report_requires_exactly_two_strategies(portfolio_returns):
    single_strategy = portfolio_returns[["Equal Weight"]]

    with pytest.raises(ValueError, match="Exactly two strategies"):
        build_backtest_report(single_strategy)


def test_report_rejects_invalid_returns():
    invalid_returns = pd.DataFrame(
        {"A": [0.01, np.nan], "B": [0.02, 0.01]}
    )

    with pytest.raises(ValueError, match="finite"):
        build_backtest_report(invalid_returns)


def test_report_identifies_smaller_maximum_drawdown(portfolio_returns):
    report = build_backtest_report(portfolio_returns)

    assert (
        report["interpretation"]["Maximum Drawdown"]
        == "Minimum Variance"
    )
