import numpy as np
import pandas as pd
import pytest

from core.returns import calculate_returns


def test_simple_returns():
    prices = pd.DataFrame(
        {
            "ASSET_A": [100.0, 110.0, 121.0],
        }
    )

    returns = calculate_returns(prices)

    expected = np.array([0.10, 0.10])

    np.testing.assert_allclose(
        returns["ASSET_A"].values,
        expected,
    )


def test_log_returns():
    prices = pd.DataFrame(
        {
            "ASSET_A": [100.0, 110.0],
        }
    )

    returns = calculate_returns(prices, method="log")

    expected = np.log(1.10)

    np.testing.assert_allclose(
        returns["ASSET_A"].iloc[0],
        expected,
    )


def test_empty_prices():
    prices = pd.DataFrame()

    with pytest.raises(ValueError):
        calculate_returns(prices)


def test_negative_prices():
    prices = pd.DataFrame(
        {
            "ASSET_A": [100.0, -110.0],
        }
    )

    with pytest.raises(ValueError):
        calculate_returns(prices)