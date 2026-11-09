import numpy as np
import pandas as pd
import pytest

from core.robustness import (
    BENCHMARK,
    run_robustness_analysis,
)


@pytest.fixture
def prices():
    rng = np.random.default_rng(42)

    dates = pd.bdate_range("2021-01-01", periods=700)

    daily_returns = rng.normal(
        loc=0.0002,
        scale=0.01,
        size=(700, 4),
    )

    values = 100 * np.cumprod(1 + daily_returns, axis=0)

    return pd.DataFrame(
        values,
        index=dates,
        columns=["Asset A", "Asset B", "Asset C", "Asset D"],
    )


def test_returns_include_buy_and_hold_benchmark(prices):
    results = run_robustness_analysis(prices)

    assert BENCHMARK in results["returns"].columns
    assert len(results["returns"]) == len(prices) - 252


def test_all_strategies_share_same_evaluation_dates(prices):
    results = run_robustness_analysis(prices)

    returns = results["returns"]

    assert returns.notna().all().all()
    assert returns.index.equals(returns.dropna().index)


def test_full_period_metrics_include_all_strategies(prices):
    results = run_robustness_analysis(prices)

    metrics = results["full_period_metrics"]

    expected = {
        "Equal Weight",
        "Minimum Variance",
        "Equal Weight Net",
        "Minimum Variance Net",
        BENCHMARK,
    }

    assert set(metrics.index) == expected
    assert np.isfinite(metrics.to_numpy()).all()


def test_annual_metrics_have_year_and_strategy_index(prices):
    results = run_robustness_analysis(prices)

    annual = results["annual_metrics"]

    assert annual.index.names == ["Year", "Strategy"]
    assert set(annual.index.get_level_values("Strategy")) == {
        "Equal Weight",
        "Minimum Variance",
        "Equal Weight Net",
        "Minimum Variance Net",
        BENCHMARK,
    }


def test_insufficient_prices_are_rejected():
    prices = pd.DataFrame(
        {"Asset A": np.arange(1, 253), "Asset B": np.arange(2, 254)},
        index=pd.bdate_range("2020-01-01", periods=252),
    )

    with pytest.raises(ValueError, match="Insufficient"):
        run_robustness_analysis(prices)


def test_buy_and_hold_benchmark_is_not_rebalanced(prices):
    results = run_robustness_analysis(prices)

    returns = results["returns"]
    benchmark = returns[BENCHMARK]

    asset_returns = prices.iloc[251:].pct_change().dropna()
    equal_weights = np.full(4, 0.25)

    asset_growth = (1 + asset_returns).cumprod()
    portfolio_value = asset_growth @ equal_weights
    expected = portfolio_value / portfolio_value.shift(1, fill_value=1.0) - 1

    expected.index.name = "Date"

    pd.testing.assert_series_equal(
        benchmark,
        expected.rename(BENCHMARK),
    )