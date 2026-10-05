import numpy as np
import pandas as pd
import pytest

from core.portfolio import (
    analyze_portfolio,
    annualized_return,
    annualized_volatility,
    portfolio_return,
)


def test_portfolio_return():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02],
            "B": [0.03, 0.04],
        }
    )

    weights = np.array([0.5, 0.5])

    result = portfolio_return(returns, weights)

    expected = np.array([0.02, 0.03])

    np.testing.assert_allclose(
        result.values,
        expected,
    )


def test_weights_must_sum_to_one():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02],
            "B": [0.03, 0.04],
        }
    )

    weights = np.array([0.3, 0.3])

    with pytest.raises(ValueError):
        portfolio_return(returns, weights)


def test_annualized_return():
    returns = pd.Series([0.01] * 252)

    result = annualized_return(returns)

    assert result == pytest.approx(0.01 * 252)


def test_annualized_volatility():
    returns = pd.Series([0.01, -0.01] * 126)

    result = annualized_volatility(returns)

    assert result > 0


def test_analyze_portfolio():
    prices = pd.DataFrame(
        {
            "A": [100, 101, 102, 103, 104],
            "B": [100, 99, 101, 100, 102],
        }
    )

    weights = np.array([0.6, 0.4])

    result = analyze_portfolio(
        prices,
        weights,
    )

    assert "returns" in result
    assert "portfolio_returns" in result
    assert "annualized_return" in result
    assert "annualized_volatility" in result
    assert "covariance" in result
    assert "risk_contributions" in result

    assert result["annualized_volatility"] > 0

    assert result["risk_contributions"].sum() == pytest.approx(
        1.0
    )


def test_sharpe_ratio():
    from core.portfolio import sharpe_ratio

    result = sharpe_ratio(
        annualized_return_value=0.08,
        annualized_volatility_value=0.16,
        risk_free_rate=0.02,
    )

    assert result == pytest.approx(0.375)