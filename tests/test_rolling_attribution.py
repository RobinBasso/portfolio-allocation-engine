
import numpy as np
import pandas as pd
import pytest

from backtesting.attribution import (
    _build_daily_weights,
    calculate_daily_attribution,
    run_rolling_attribution,
)


@pytest.fixture
def asset_returns():
    return pd.DataFrame(
        {
            "Asset A": [0.02, -0.01, 0.03],
            "Asset B": [0.01, 0.02, -0.01],
        },
        index=pd.date_range("2025-01-01", periods=3),
    )


@pytest.fixture
def portfolio_weights():
    return pd.DataFrame(
        {
            "Asset A": [0.60, 0.61, 0.59],
            "Asset B": [0.40, 0.39, 0.41],
        },
        index=pd.date_range("2025-01-01", periods=3),
    )


def test_daily_weights_drift_between_rebalances():

    dates = pd.bdate_range("2025-01-01", periods=3)

    asset_returns = pd.DataFrame(
        {
            "Asset A": [0.10, 0.00, 0.00],
            "Asset B": [0.00, 0.10, 0.00],
        },
        index=dates,
    )

    target_weights = pd.DataFrame(
        {"Asset A": [0.50], "Asset B": [0.50]},
        index=[dates[0]],
    )

    result = _build_daily_weights(
        asset_returns,
        target_weights,
    )

    expected = pd.DataFrame(
        {
            "Asset A": [0.50, 0.5238095238095238, 0.50],
            "Asset B": [0.50, 0.47619047619047616, 0.50],
        },
        index=dates,
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
        rtol=1e-10,
        atol=1e-12,
    )


def test_returns_dataframe(asset_returns, portfolio_weights):
    result = calculate_daily_attribution(
        asset_returns,
        portfolio_weights,
    )

    assert isinstance(result, pd.DataFrame)


def test_preserves_dates(asset_returns, portfolio_weights):
    result = calculate_daily_attribution(
        asset_returns,
        portfolio_weights,
    )

    assert result.index.equals(asset_returns.index)


def test_calculates_asset_contributions(asset_returns, portfolio_weights):
    result = calculate_daily_attribution(
        asset_returns,
        portfolio_weights,
    )

    expected = asset_returns * portfolio_weights

    pd.testing.assert_frame_equal(
        result[["Asset A", "Asset B"]],
        expected,
    )


def test_gross_returns_equal_sum_of_contributions(
    asset_returns,
    portfolio_weights,
):
    result = calculate_daily_attribution(
        asset_returns,
        portfolio_weights,
    )

    expected = (
        asset_returns * portfolio_weights
    ).sum(axis=1)

    pd.testing.assert_series_equal(
        result["Gross Portfolio Return"],
        expected,
        check_names=False,
    )


def test_zero_transaction_costs_by_default(
    asset_returns,
    portfolio_weights,
):
    result = calculate_daily_attribution(
        asset_returns,
        portfolio_weights,
    )

    assert (result["Transaction Cost"] == 0.0).all()


def test_net_returns_equal_gross_without_costs(
    asset_returns,
    portfolio_weights,
):
    result = calculate_daily_attribution(
        asset_returns,
        portfolio_weights,
    )

    pd.testing.assert_series_equal(
        result["Net Portfolio Return"],
        result["Gross Portfolio Return"],
        check_names=False,
    )


def test_transaction_costs_are_deducted(
    asset_returns,
    portfolio_weights,
):
    costs = pd.Series(
        [0.001, 0.0, 0.002],
        index=asset_returns.index,
    )

    result = calculate_daily_attribution(
        asset_returns,
        portfolio_weights,
        costs,
    )

    expected = (
        result["Gross Portfolio Return"] - costs
    )

    pd.testing.assert_series_equal(
        result["Net Portfolio Return"],
        expected,
        check_names=False,
    )


def test_transaction_costs_are_not_attributed_to_assets(
    asset_returns,
    portfolio_weights,
):
    costs = pd.Series(
        [0.001, 0.0, 0.002],
        index=asset_returns.index,
    )

    result = calculate_daily_attribution(
        asset_returns,
        portfolio_weights,
        costs,
    )

    expected = asset_returns * portfolio_weights

    pd.testing.assert_frame_equal(
        result[["Asset A", "Asset B"]],
        expected,
    )


def test_rejects_non_dataframe_returns(portfolio_weights):
    with pytest.raises(TypeError):
        calculate_daily_attribution(
            np.array([[0.01, 0.02]]),
            portfolio_weights,
        )


def test_rejects_empty_returns(portfolio_weights):
    with pytest.raises(ValueError):
        calculate_daily_attribution(
            pd.DataFrame(),
            portfolio_weights,
        )


def test_rejects_mismatched_dates(asset_returns, portfolio_weights):
    weights = portfolio_weights.copy()
    weights.index = weights.index + pd.Timedelta(days=1)

    with pytest.raises(ValueError):
        calculate_daily_attribution(asset_returns, weights)


def test_rejects_mismatched_assets(asset_returns, portfolio_weights):
    weights = portfolio_weights.rename(
        columns={"Asset B": "Asset C"}
    )

    with pytest.raises(ValueError):
        calculate_daily_attribution(asset_returns, weights)


def test_rejects_weights_not_summing_to_one(
    asset_returns,
    portfolio_weights,
):
    weights = portfolio_weights.copy()
    weights.iloc[0, 0] = 0.50

    with pytest.raises(ValueError):
        calculate_daily_attribution(asset_returns, weights)


def test_rejects_negative_weights(asset_returns, portfolio_weights):
    weights = portfolio_weights.copy()
    weights.iloc[0, 0] = 1.10
    weights.iloc[0, 1] = -0.10

    with pytest.raises(ValueError):
        calculate_daily_attribution(asset_returns, weights)


def test_rejects_returns_below_minus_100_percent(
    portfolio_weights,
):
    returns = pd.DataFrame(
        {
            "Asset A": [-1.10, 0.01, 0.02],
            "Asset B": [0.01, 0.02, 0.01],
        },
        index=portfolio_weights.index,
    )

    with pytest.raises(ValueError):
        calculate_daily_attribution(returns, portfolio_weights)


def test_rejects_negative_transaction_costs(
    asset_returns,
    portfolio_weights,
):
    costs = pd.Series(
        [0.001, -0.001, 0.0],
        index=asset_returns.index,
    )

    with pytest.raises(ValueError):
        calculate_daily_attribution(
            asset_returns,
            portfolio_weights,
            costs,
        )


def test_rejects_misaligned_transaction_costs(
    asset_returns,
    portfolio_weights,
):
    costs = pd.Series(
        [0.001, 0.0, 0.002],
        index=asset_returns.index + pd.Timedelta(days=1),
    )

    with pytest.raises(ValueError):
        calculate_daily_attribution(
            asset_returns,
            portfolio_weights,
            costs,
        )



@pytest.fixture
def rolling_prices():
    rng = np.random.default_rng(42)

    dates = pd.bdate_range("2020-01-01", periods=400)
    assets = ["US_EQUITY", "GOV_BONDS", "GOLD"]

    daily_returns = rng.normal(
        loc=[0.0003, 0.0001, 0.0002],
        scale=[0.015, 0.004, 0.009],
        size=(len(dates) - 1, len(assets)),
    )

    values = 100 * np.vstack([
        np.ones(len(assets)),
        np.cumprod(1 + daily_returns, axis=0),
    ])

    return pd.DataFrame(
        values,
        index=dates,
        columns=assets,
    )


def test_rolling_attribution_returns_both_strategies(rolling_prices):
    result = run_rolling_attribution(
        rolling_prices,
        estimation_window=100,
        rebalance_frequency=20,
    )

    assert set(result) == {"Equal Weight", "Minimum Variance"}


def test_rolling_attribution_matches_backtest_dates(rolling_prices):
    from backtesting.rolling import run_rolling_backtest

    backtest = run_rolling_backtest(
        rolling_prices,
        estimation_window=100,
        rebalance_frequency=20,
    )

    attribution = run_rolling_attribution(
        rolling_prices,
        estimation_window=100,
        rebalance_frequency=20,
    )

    for result in attribution.values():
        assert result.index.equals(backtest.index)


@pytest.mark.parametrize(
    "strategy, gross_column",
    [
        ("Equal Weight", "Equal Weight"),
        ("Minimum Variance", "Minimum Variance"),
    ],
)
def test_attribution_gross_returns_match_backtest(
    rolling_prices,
    strategy,
    gross_column,
):
    from backtesting.rolling import run_rolling_backtest

    backtest = run_rolling_backtest(
        rolling_prices,
        estimation_window=100,
        rebalance_frequency=20,
    )

    attribution = run_rolling_attribution(
        rolling_prices,
        estimation_window=100,
        rebalance_frequency=20,
    )[strategy]

    np.testing.assert_allclose(
        attribution["Gross Portfolio Return"].to_numpy(),
        backtest[gross_column].to_numpy(),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "strategy, net_column",
    [
        ("Equal Weight", "Equal Weight Net"),
        ("Minimum Variance", "Minimum Variance Net"),
    ],
)
def test_attribution_net_returns_match_backtest(
    rolling_prices,
    strategy,
    net_column,
):
    from backtesting.rolling import run_rolling_backtest

    backtest = run_rolling_backtest(
        rolling_prices,
        estimation_window=100,
        rebalance_frequency=20,
        transaction_cost_bps=10.0,
    )

    attribution = run_rolling_attribution(
        rolling_prices,
        estimation_window=100,
        rebalance_frequency=20,
        transaction_cost_bps=10.0,
    )[strategy]

    np.testing.assert_allclose(
        attribution["Net Portfolio Return"].to_numpy(),
        backtest[net_column].to_numpy(),
        rtol=1e-10,
        atol=1e-12,
    )


def test_attribution_costs_are_only_applied_at_rebalances(rolling_prices):
    attribution = run_rolling_attribution(
        rolling_prices,
        estimation_window=100,
        rebalance_frequency=20,
        transaction_cost_bps=10.0,
    )

    expected_dates = attribution["Minimum Variance"].index[20::20]

    for result in attribution.values():
        cost_dates = result.index[
            result["Transaction Cost"] > 0.0
        ]

        assert cost_dates.equals(expected_dates)


def test_zero_transaction_costs_preserve_gross_returns(rolling_prices):
    attribution = run_rolling_attribution(
        rolling_prices,
        estimation_window=100,
        rebalance_frequency=20,
        transaction_cost_bps=0.0,
    )

    for result in attribution.values():
        np.testing.assert_allclose(
            result["Gross Portfolio Return"].to_numpy(),
            result["Net Portfolio Return"].to_numpy(),
        )

        assert (result["Transaction Cost"] == 0.0).all()


def test_daily_weights_reject_reordered_assets():
    dates = pd.bdate_range("2025-01-01", periods=2)

    asset_returns = pd.DataFrame(
        {
            "Asset A": [0.01, 0.02],
            "Asset B": [0.02, 0.01],
        },
        index=dates,
    )

    target_weights = pd.DataFrame(
        {
            "Asset B": [0.50],
            "Asset A": [0.50],
        },
        index=[dates[0]],
    )

    with pytest.raises(ValueError, match="same order"):
        _build_daily_weights(asset_returns, target_weights)


def test_daily_weights_reject_invalid_target_weights():
    dates = pd.bdate_range("2025-01-01", periods=2)

    asset_returns = pd.DataFrame(
        {
            "Asset A": [0.01, 0.02],
            "Asset B": [0.02, 0.01],
        },
        index=dates,
    )

    target_weights = pd.DataFrame(
        {
            "Asset A": [0.60],
            "Asset B": [0.30],
        },
        index=[dates[0]],
    )

    with pytest.raises(ValueError, match="sum to 1"):
        _build_daily_weights(asset_returns, target_weights)