
import numpy as np
import pandas as pd
import pytest

from core.reporting import build_allocation_report


@pytest.fixture
def prices():
    rng = np.random.default_rng(42)
    daily_returns = rng.normal(
        loc=0.0003,
        scale=0.01,
        size=(500, 3),
    )
    return pd.DataFrame(
        100 * np.cumprod(1 + daily_returns, axis=0),
        columns=["A", "B", "C"],
    )


@pytest.fixture
def weights():
    return pd.Series(
        {"A": 0.50, "B": 0.30, "C": 0.20}
    )


def test_report_contains_expected_outputs(prices, weights):
    report = build_allocation_report(prices, weights)

    assert set(report) == {"portfolio_metrics", "asset_report"}
    assert list(report["portfolio_metrics"].index) == [
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
    ]
    assert list(report["asset_report"].columns) == [
        "Allocation",
        "Risk Contribution",
        "Risk-to-Weight Ratio",
    ]


def test_allocation_sums_to_one(prices, weights):
    report = build_allocation_report(prices, weights)

    assert report["asset_report"]["Allocation"].sum() == pytest.approx(1.0)


def test_risk_contributions_sum_to_one(prices, weights):
    report = build_allocation_report(prices, weights)

    assert (
        report["asset_report"]["Risk Contribution"].sum()
        == pytest.approx(1.0)
    )


def test_portfolio_metrics_are_finite(prices, weights):
    report = build_allocation_report(prices, weights)

    assert np.isfinite(report["portfolio_metrics"].to_numpy()).all()


def test_weights_must_sum_to_one(prices):
    invalid_weights = pd.Series(
        {"A": 0.50, "B": 0.30, "C": 0.10}
    )

    with pytest.raises(ValueError, match="sum to one"):
        build_allocation_report(prices, invalid_weights)


def test_weight_labels_must_match_assets(prices):
    invalid_weights = pd.Series(
        {"A": 0.50, "B": 0.30, "D": 0.20}
    )

    with pytest.raises(ValueError, match="labels must match"):
        build_allocation_report(prices, invalid_weights)


def test_negative_weights_are_rejected(prices):
    invalid_weights = pd.Series(
        {"A": 0.60, "B": 0.50, "C": -0.10}
    )

    with pytest.raises(ValueError, match="non-negative"):
        build_allocation_report(prices, invalid_weights)
