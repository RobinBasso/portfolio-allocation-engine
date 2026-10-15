
import numpy as np
import pandas as pd
import pytest

from scenarios.historical import historical_stress_report


@pytest.fixture
def prices():
    rng = np.random.default_rng(42)
    daily_returns = rng.normal(
        loc=0.0002,
        scale=0.01,
        size=(100, 2),
    )
    return pd.DataFrame(
        100 * np.cumprod(1 + daily_returns, axis=0),
        columns=["A", "B"],
    )


@pytest.fixture
def portfolios():
    return {
        "Equal Weight": pd.Series({"A": 0.5, "B": 0.5}),
        "Portfolio A": pd.Series({"A": 1.0, "B": 0.0}),
    }


def test_report_contains_expected_metrics(prices, portfolios):
    report = historical_stress_report(prices, portfolios)

    assert list(report.index) == ["Equal Weight", "Portfolio A"]
    assert list(report.columns) == [
        "Maximum Drawdown",
        "Worst 21-Day Return",
        "Worst Daily Return",
    ]


def test_drawdown_is_non_positive(prices, portfolios):
    report = historical_stress_report(prices, portfolios)

    assert (report["Maximum Drawdown"] <= 0).all()


def test_stress_metrics_are_finite(prices, portfolios):
    report = historical_stress_report(prices, portfolios)

    assert np.isfinite(report.to_numpy()).all()


def test_custom_window_changes_metric_name(prices, portfolios):
    report = historical_stress_report(prices, portfolios, window=10)

    assert "Worst 10-Day Return" in report.columns


def test_empty_prices_raise_error(portfolios):
    with pytest.raises(ValueError, match="Price data cannot be empty"):
        historical_stress_report(pd.DataFrame(), portfolios)


def test_invalid_window_raises_error(prices, portfolios):
    with pytest.raises(ValueError, match="positive integer"):
        historical_stress_report(prices, portfolios, window=0)


def test_weights_must_sum_to_one(prices):
    invalid_portfolios = {
        "Invalid": pd.Series({"A": 0.6, "B": 0.3})
    }

    with pytest.raises(ValueError, match="sum to one"):
        historical_stress_report(prices, invalid_portfolios)
