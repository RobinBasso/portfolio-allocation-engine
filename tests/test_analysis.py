
import numpy as np
import pandas as pd
import pytest

from core.analysis import run_portfolio_analysis


@pytest.fixture
def prices():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=300)
    assets = ["US_EQUITY", "GOV_BONDS", "GOLD"]

    daily_returns = rng.normal(
        loc=[0.0003, 0.0001, 0.0002],
        scale=[0.015, 0.004, 0.009],
        size=(len(dates), len(assets)),
    )

    return pd.DataFrame(
        100 * np.vstack(
            [np.ones(len(assets)), np.cumprod(1 + daily_returns, axis=0)]
        )[1:],
        index=dates,
        columns=assets,
    )


def test_returns_all_analysis_sections(prices):
    result = run_portfolio_analysis(prices)

    assert set(result) == {
        "comparison",
        "allocations",
        "risk_reports",
        "historical_stress",
        "parametric_stress",
    }


def test_constructs_fully_invested_portfolios(prices):
    result = run_portfolio_analysis(prices)

    for weights in result["allocations"].values():
        assert weights.sum() == pytest.approx(1.0)
        assert (weights >= 0).all()


def test_reports_both_portfolios(prices):
    result = run_portfolio_analysis(prices)

    assert set(result["allocations"]) == {
        "Equal Weight",
        "Minimum Variance",
    }
    assert set(result["risk_reports"]) == {
        "Equal Weight",
        "Minimum Variance",
    }


def test_stress_reports_include_both_portfolios(prices):
    result = run_portfolio_analysis(prices)

    assert set(result["historical_stress"].index) == {
        "Equal Weight",
        "Minimum Variance",
    }
    assert set(result["parametric_stress"].index) == {
        "Equal Weight",
        "Minimum Variance",
    }


def test_rejects_empty_prices():
    with pytest.raises(ValueError, match="at least one asset"):
        run_portfolio_analysis(pd.DataFrame())


def test_rejects_duplicate_asset_labels(prices):
    prices.columns = ["US_EQUITY", "GOV_BONDS", "US_EQUITY"]

    with pytest.raises(ValueError, match="unique"):
        run_portfolio_analysis(prices)


def test_rejects_invalid_stress_window(prices):
    with pytest.raises(ValueError, match="[Ww]indow"):
        run_portfolio_analysis(prices, stress_window=0)
