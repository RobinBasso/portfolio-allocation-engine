
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


def test_analysis_recalculates_metrics_for_shorter_estimation_period(prices):
    full_history = run_portfolio_analysis(prices)
    recent_history = run_portfolio_analysis(prices.tail(150))

    full_metrics = full_history["comparison"]
    recent_metrics = recent_history["comparison"]

    assert not np.allclose(
        full_metrics["Annualized Return"].values,
        recent_metrics["Annualized Return"].values,
    )
    assert not np.allclose(
        full_metrics["Annualized Volatility"].values,
        recent_metrics["Annualized Volatility"].values,
    )

def test_sample_covariance_is_the_default_method(prices):
    default_result = run_portfolio_analysis(prices)
    sample_result = run_portfolio_analysis(
        prices,
        covariance_method="sample",
    )

    pd.testing.assert_series_equal(
        default_result["allocations"]["Minimum Variance"],
        sample_result["allocations"]["Minimum Variance"],
    )


def test_shrinkage_method_uses_shrinkage_covariance(prices, monkeypatch):
    import core.analysis as analysis

    original_function = analysis.calculate_shrinkage_covariance
    calls = []

    def tracked_function(returns, shrinkage=0.20, annualize=True):
        calls.append(shrinkage)
        return original_function(
            returns,
            shrinkage=shrinkage,
            annualize=annualize,
        )

    monkeypatch.setattr(
        analysis,
        "calculate_shrinkage_covariance",
        tracked_function,
    )

    result = run_portfolio_analysis(
        prices,
        covariance_method="shrinkage",
        shrinkage=0.50,
    )

    weights = result["allocations"]["Minimum Variance"]

    assert calls == [0.50]
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()


def test_rejects_unsupported_covariance_method(prices):
    with pytest.raises(ValueError, match="Covariance method"):
        run_portfolio_analysis(
            prices,
            covariance_method="unsupported",
        )


def test_ledoit_wolf_method_uses_ledoit_wolf_covariance(prices, monkeypatch):

    import core.analysis as analysis

    original_function = analysis.calculate_ledoit_wolf_covariance

    calls = []

    def tracked_function(returns, annualize=True):
        calls.append(True)
        return original_function(
            returns,
            annualize=annualize,
        )

    monkeypatch.setattr(
        analysis,
        "calculate_ledoit_wolf_covariance",
        tracked_function,
    )

    result = run_portfolio_analysis(
        prices,
        covariance_method="ledoit_wolf",
    )

    weights = result["allocations"]["Minimum Variance"]

    assert calls == [True]
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
