import numpy as np
import pandas as pd
import pytest

from core.portfolio import (
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

    np.testing.assert_allclose(result.values, expected)


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