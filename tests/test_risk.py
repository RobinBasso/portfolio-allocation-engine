import numpy as np
import pandas as pd
import pytest

from core.covariance import calculate_covariance
from core.risk import (
    component_contribution_to_risk,
    marginal_contribution_to_risk,
    percentage_contribution_to_risk,
    portfolio_volatility,
)


@pytest.fixture
def covariance_matrix():
    return pd.DataFrame(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ],
        columns=["A", "B"],
        index=["A", "B"],
    )


@pytest.fixture
def weights():
    return np.array([0.6, 0.4])


def test_portfolio_volatility(covariance_matrix, weights):
    result = portfolio_volatility(
        covariance_matrix,
        weights,
    )

    expected_variance = (
        0.6**2 * 0.04
        + 0.4**2 * 0.09
        + 2 * 0.6 * 0.4 * 0.01
    )

    expected = np.sqrt(expected_variance)

    assert result == pytest.approx(expected)


def test_marginal_risk_contribution(
    covariance_matrix,
    weights,
):
    result = marginal_contribution_to_risk(
        covariance_matrix,
        weights,
    )

    portfolio_risk = portfolio_volatility(
        covariance_matrix,
        weights,
    )

    expected = np.array([
        (0.04 * 0.6 + 0.01 * 0.4) / portfolio_risk,
        (0.01 * 0.6 + 0.09 * 0.4) / portfolio_risk,
    ])

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_component_contributions_sum_to_total_risk(
    covariance_matrix,
    weights,
):
    components = component_contribution_to_risk(
        covariance_matrix,
        weights,
    )

    portfolio_risk = portfolio_volatility(
        covariance_matrix,
        weights,
    )

    assert components.sum() == pytest.approx(
        portfolio_risk
    )


def test_percentage_contributions_sum_to_one(
    covariance_matrix,
    weights,
):
    contributions = percentage_contribution_to_risk(
        covariance_matrix,
        weights,
    )

    assert contributions.sum() == pytest.approx(1.0)


def test_covariance_annualization():
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.01, 0.02],
            "B": [0.02, 0.00, -0.01],
        }
    )

    daily_covariance = calculate_covariance(
        returns,
        annualize=False,
    )

    annual_covariance = calculate_covariance(
        returns,
        annualize=True,
    )

    pd.testing.assert_frame_equal(
        annual_covariance,
        daily_covariance * 252,
    )