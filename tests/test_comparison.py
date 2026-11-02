
import numpy as np
import pandas as pd
import pytest

from core.comparison import compare_portfolios


@pytest.fixture
def prices():
    rng = np.random.default_rng(42)
    n_days = 500

    daily_returns = rng.normal(
        loc=0.0003,
        scale=0.01,
        size=(n_days, 3),
    )

    return pd.DataFrame(
        100 * np.cumprod(1 + daily_returns, axis=0),
        columns=["A", "B", "C"],
    )


def test_comparison_returns_expected_portfolios(prices):
    result = compare_portfolios(prices)

    assert list(result.index) == [
        "Equal Weight",
        "Minimum Variance",
    ]


def test_comparison_returns_expected_metrics(prices):
    result = compare_portfolios(prices)

    assert list(result.columns) == [
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
    ]


def test_volatility_is_positive(prices):
    result = compare_portfolios(prices)

    assert (result["Annualized Volatility"] > 0).all()


def test_metrics_are_finite(prices):
    result = compare_portfolios(prices)

    assert np.isfinite(result.to_numpy()).all()


def test_empty_prices_raise_error():
    with pytest.raises(ValueError, match="Price data cannot be empty"):
        compare_portfolios(pd.DataFrame())


def test_sharpe_ratio_changes_with_risk_free_rate(prices):
    zero_rate = compare_portfolios(prices, risk_free_rate=0.0)
    higher_rate = compare_portfolios(prices, risk_free_rate=0.03)

    assert (
        higher_rate["Sharpe Ratio"] < zero_rate["Sharpe Ratio"]
    ).all()

    pd.testing.assert_series_equal(
        zero_rate["Annualized Return"],
        higher_rate["Annualized Return"],
    )

    pd.testing.assert_series_equal(
        zero_rate["Annualized Volatility"],
        higher_rate["Annualized Volatility"],
    )