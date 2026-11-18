
import numpy as np
import pandas as pd
import pytest

from core.attribution import calculate_buy_and_hold_attribution


@pytest.fixture
def asset_returns():
    return pd.DataFrame(
        {
            "Asset A": [0.10, 0.05, -0.02],
            "Asset B": [0.02, 0.03, 0.04],
        }
    )


@pytest.fixture
def initial_weights():
    return pd.Series(
        {
            "Asset A": 0.60,
            "Asset B": 0.40,
        }
    )


def test_attribution_returns_dataframe(asset_returns, initial_weights):
    result = calculate_buy_and_hold_attribution(
        asset_returns,
        initial_weights,
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "initial_weight",
        "cumulative_asset_return",
        "return_contribution",
        "contribution_percentage",
    ]


def test_attribution_preserves_asset_order(asset_returns, initial_weights):
    result = calculate_buy_and_hold_attribution(
        asset_returns,
        initial_weights,
    )

    assert result.index.equals(asset_returns.columns)


def test_weights_are_preserved(asset_returns, initial_weights):
    result = calculate_buy_and_hold_attribution(
        asset_returns,
        initial_weights,
    )

    pd.testing.assert_series_equal(
        result["initial_weight"],
        initial_weights,
        check_names=False,
    )


def test_cumulative_asset_returns(asset_returns, initial_weights):
    result = calculate_buy_and_hold_attribution(
        asset_returns,
        initial_weights,
    )

    expected = (1.10 * 1.05 * 0.98) - 1.0

    assert result.loc["Asset A", "cumulative_asset_return"] == pytest.approx(
        expected
    )


def test_contributions_equal_weighted_asset_returns(
    asset_returns,
    initial_weights,
):
    result = calculate_buy_and_hold_attribution(
        asset_returns,
        initial_weights,
    )

    expected = (
        initial_weights * ((1.0 + asset_returns).prod(axis=0) - 1.0)
    )

    pd.testing.assert_series_equal(
        result["return_contribution"],
        expected,
        check_names=False,
    )


def test_contributions_sum_to_portfolio_return(
    asset_returns,
    initial_weights,
):
    result = calculate_buy_and_hold_attribution(
        asset_returns,
        initial_weights,
    )

    portfolio_return = result["return_contribution"].sum()

    assert portfolio_return == pytest.approx(
        (
            initial_weights
            * ((1.0 + asset_returns).prod(axis=0) - 1.0)
        ).sum()
    )


def test_contribution_percentages_sum_to_100(
    asset_returns,
    initial_weights,
):
    result = calculate_buy_and_hold_attribution(
        asset_returns,
        initial_weights,
    )

    assert result["contribution_percentage"].sum() == pytest.approx(100.0)


def test_zero_portfolio_return_produces_nan_percentages():
    returns = pd.DataFrame(
        {
            "Asset A": [0.10],
            "Asset B": [-0.10],
        }
    )

    weights = pd.Series(
        {
            "Asset A": 0.50,
            "Asset B": 0.50,
        }
    )

    result = calculate_buy_and_hold_attribution(returns, weights)

    assert result["return_contribution"].sum() == pytest.approx(0.0)
    assert result["contribution_percentage"].isna().all()


def test_rejects_non_dataframe_returns(initial_weights):
    with pytest.raises(TypeError):
        calculate_buy_and_hold_attribution(
            np.array([[0.01, 0.02]]),
            initial_weights,
        )


def test_rejects_empty_returns(initial_weights):
    returns = pd.DataFrame()

    with pytest.raises(ValueError):
        calculate_buy_and_hold_attribution(returns, initial_weights)


def test_rejects_mismatched_assets(asset_returns):
    weights = pd.Series(
        {
            "Asset A": 0.50,
            "Asset C": 0.50,
        }
    )

    with pytest.raises(ValueError):
        calculate_buy_and_hold_attribution(asset_returns, weights)


def test_rejects_weights_not_summing_to_one(asset_returns):
    weights = pd.Series(
        {
            "Asset A": 0.40,
            "Asset B": 0.40,
        }
    )

    with pytest.raises(ValueError):
        calculate_buy_and_hold_attribution(asset_returns, weights)


def test_rejects_negative_weights(asset_returns):
    weights = pd.Series(
        {
            "Asset A": 1.20,
            "Asset B": -0.20,
        }
    )

    with pytest.raises(ValueError):
        calculate_buy_and_hold_attribution(asset_returns, weights)


def test_rejects_returns_below_minus_100_percent(initial_weights):
    returns = pd.DataFrame(
        {
            "Asset A": [-1.10],
            "Asset B": [0.05],
        }
    )

    with pytest.raises(ValueError):
        calculate_buy_and_hold_attribution(returns, initial_weights)


def test_rejects_missing_values(initial_weights):
    returns = pd.DataFrame(
        {
            "Asset A": [0.05, np.nan],
            "Asset B": [0.02, 0.03],
        }
    )

    with pytest.raises(ValueError):
        calculate_buy_and_hold_attribution(returns, initial_weights)


def test_rejects_duplicate_asset_columns(initial_weights):
    returns = pd.DataFrame(
        [[0.05, 0.02]],
        columns=["Asset A", "Asset A"],
    )

    with pytest.raises(ValueError):
        calculate_buy_and_hold_attribution(returns, initial_weights)
