
import pandas as pd
import pytest

from scenarios.parametric import parametric_stress_report


@pytest.fixture
def portfolios():
    return {
        "Equal Weight": pd.Series(
            {"US_EQUITY": 0.5, "GOV_BONDS": 0.5}
        ),
        "Defensive": pd.Series(
            {"US_EQUITY": 0.2, "GOV_BONDS": 0.8}
        ),
    }


def test_calculates_weighted_scenario_returns(portfolios):
    scenarios = {
        "Sell-Off": {"US_EQUITY": -0.20, "GOV_BONDS": 0.05}
    }

    report = parametric_stress_report(portfolios, scenarios)

    assert report.loc["Equal Weight", "Sell-Off"] == pytest.approx(-0.075)
    assert report.loc["Defensive", "Sell-Off"] == pytest.approx(0.0)


def test_uses_default_scenarios(portfolios):
    report = parametric_stress_report(portfolios)

    assert "Equity Sell-Off" in report.columns
    assert "Bond Sell-Off" in report.columns
    assert "Combined Risk-Off" in report.columns


def test_rejects_weights_that_do_not_sum_to_one():
    portfolios = {
        "Invalid": pd.Series({"US_EQUITY": 0.4, "GOV_BONDS": 0.4})
    }

    with pytest.raises(ValueError, match="sum to 1"):
        parametric_stress_report(
            portfolios,
            {"Shock": {"US_EQUITY": -0.1, "GOV_BONDS": 0.0}},
        )


def test_rejects_missing_asset_shocks(portfolios):
    scenarios = {"Incomplete": {"US_EQUITY": -0.20}}

    with pytest.raises(ValueError, match="missing shocks"):
        parametric_stress_report(portfolios, scenarios)


def test_rejects_negative_weights():
    portfolios = {
        "Invalid": pd.Series({"US_EQUITY": 1.2, "GOV_BONDS": -0.2})
    }

    with pytest.raises(ValueError, match="non-negative"):
        parametric_stress_report(
            portfolios,
            {"Shock": {"US_EQUITY": -0.1, "GOV_BONDS": 0.0}},
        )


def test_rejects_non_finite_shocks(portfolios):
    scenarios = {
        "Invalid": {"US_EQUITY": float("nan"), "GOV_BONDS": 0.0}
    }

    with pytest.raises(ValueError, match="finite"):
        parametric_stress_report(portfolios, scenarios)


def test_rejects_empty_inputs():
    with pytest.raises(ValueError, match="portfolio"):
        parametric_stress_report({})

    with pytest.raises(ValueError, match="scenario"):
        parametric_stress_report(
            {"Equal Weight": pd.Series({"US_EQUITY": 1.0})},
            {},
        )
