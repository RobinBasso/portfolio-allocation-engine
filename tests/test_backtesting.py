import numpy as np
import pandas as pd
import pytest

from backtesting.rolling import run_rolling_backtest
from core.covariance import calculate_covariance
from core.returns import calculate_returns
from optimization.minimum_variance import minimum_variance_weights


@pytest.fixture
def prices():
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

    return pd.DataFrame(values, index=dates, columns=assets)


def test_returns_both_portfolio_strategies(prices):
    result = run_rolling_backtest(prices)

    assert list(result.columns) == [
        "Equal Weight",
        "Minimum Variance",
        "Equal Weight Net",
        "Minimum Variance Net",
        "Equal Weight Turnover",
        "Minimum Variance Turnover",
    ]


def test_backtest_starts_after_estimation_window(prices):
    result = run_rolling_backtest(
        prices,
        estimation_window=100,
        rebalance_frequency=20,
    )

    assert result.index[0] == prices.index[100]


def test_backtest_has_expected_number_of_observations(prices):
    result = run_rolling_backtest(
        prices,
        estimation_window=100,
        rebalance_frequency=20,
    )

    assert len(result) == len(prices) - 100


def test_backtest_dates_are_unique_and_ascending(prices):
    result = run_rolling_backtest(prices)

    assert result.index.is_unique
    assert result.index.is_monotonic_increasing


def test_first_period_uses_only_training_data(prices):
    estimation_window = 100
    rebalance_frequency = 20

    result = run_rolling_backtest(
        prices,
        estimation_window=estimation_window,
        rebalance_frequency=rebalance_frequency,
    )

    training_prices = prices.iloc[:estimation_window]
    training_returns = calculate_returns(training_prices)
    covariance = calculate_covariance(
        training_returns,
        annualize=False,
    )
    weights = minimum_variance_weights(covariance)

    first_test_prices = prices.iloc[
        estimation_window - 1:
        estimation_window + rebalance_frequency
    ]
    first_test_returns = calculate_returns(first_test_prices)

    asset_growth = (1 + first_test_returns).cumprod()
    portfolio_value = asset_growth @ weights
    expected_returns = (
        portfolio_value / portfolio_value.shift(1, fill_value=1.0) - 1
    )

    pd.testing.assert_series_equal(
        result["Minimum Variance"].iloc[:rebalance_frequency],
        expected_returns,
        check_names=False,
    )


def test_equal_weight_returns_are_correct(prices):
    estimation_window = 100
    rebalance_frequency = 20
    result = run_rolling_backtest(
        prices,
        estimation_window=estimation_window,
        rebalance_frequency=rebalance_frequency,
    )

    equal_weights = np.full(prices.shape[1], 1 / prices.shape[1])
    expected_periods = []

    for train_end in range(
        estimation_window,
        len(prices),
        rebalance_frequency,
    ):
        test_end = min(train_end + rebalance_frequency, len(prices))
        test_prices = prices.iloc[train_end - 1:test_end]
        asset_returns = calculate_returns(test_prices)

        asset_growth = (1 + asset_returns).cumprod()
        portfolio_value = asset_growth @ equal_weights
        period_returns = (
            portfolio_value / portfolio_value.shift(1, fill_value=1.0) - 1
        )

        expected_periods.append(period_returns)

    expected_returns = pd.concat(expected_periods)

    pd.testing.assert_series_equal(
        result["Equal Weight"],
        expected_returns,
        check_names=False,
    )


def test_rejects_insufficient_price_history(prices):
    with pytest.raises(ValueError, match="Insufficient"):
        run_rolling_backtest(
            prices.iloc[:100],
            estimation_window=100,
        )


def test_rejects_invalid_estimation_window(prices):
    with pytest.raises(ValueError, match="Estimation window"):
        run_rolling_backtest(prices, estimation_window=1)


def test_rejects_invalid_rebalance_frequency(prices):
    with pytest.raises(ValueError, match="Rebalancing frequency"):
        run_rolling_backtest(prices, rebalance_frequency=0)


def test_rejects_duplicate_asset_columns(prices):
    prices.columns = ["A", "B", "A"]

    with pytest.raises(ValueError, match="unique"):
        run_rolling_backtest(prices)


def test_rejects_unsorted_dates(prices):
    prices = prices.iloc[::-1]

    with pytest.raises(ValueError, match="ascending"):
        run_rolling_backtest(prices)


def test_first_backtest_period_has_zero_turnover(prices):
    result = run_rolling_backtest(prices)

    first_date = result.index[0]

    assert result.loc[first_date, "Equal Weight Turnover"] == 0.0
    assert result.loc[first_date, "Minimum Variance Turnover"] == 0.0


def test_turnover_is_only_recorded_at_rebalancing_dates(prices):
    result = run_rolling_backtest(
        prices,
        estimation_window=100,
        rebalance_frequency=20,
    )

    turnover_columns = [
        "Equal Weight Turnover",
        "Minimum Variance Turnover",
    ]

    for column in turnover_columns:
        nonzero_dates = result.index[result[column] > 0]

        expected_dates = result.index[20::20]

        assert nonzero_dates.equals(expected_dates)



def test_zero_transaction_costs_preserve_gross_returns(prices):
    result = run_rolling_backtest(
        prices,
        transaction_cost_bps=0.0,
    )

    np.testing.assert_allclose(
        result["Equal Weight"].to_numpy(),
        result["Equal Weight Net"].to_numpy(),
    )

    np.testing.assert_allclose(
        result["Minimum Variance"].to_numpy(),
        result["Minimum Variance Net"].to_numpy(),
    )


def test_transaction_costs_reduce_net_returns(prices):
    result = run_rolling_backtest(
        prices,
        transaction_cost_bps=10.0,
    )

    assert (
        result["Equal Weight Net"] <= result["Equal Weight"]
    ).all()

    assert (
        result["Minimum Variance Net"] <= result["Minimum Variance"]
    ).all()


def test_transaction_costs_are_applied_only_at_rebalancing_dates(prices):
    result = run_rolling_backtest(
        prices,
        estimation_window=100,
        rebalance_frequency=20,
        transaction_cost_bps=10.0,
    )

    equal_cost_dates = result.index[
        result["Equal Weight Net"] < result["Equal Weight"]
    ]

    minimum_variance_cost_dates = result.index[
        result["Minimum Variance Net"] < result["Minimum Variance"]
    ]

    expected_dates = result.index[20::20]

    assert equal_cost_dates.equals(expected_dates)
    assert minimum_variance_cost_dates.equals(expected_dates)


def test_rejects_invalid_transaction_costs(prices):
    import pytest

    for invalid_cost in [-1.0, 10_000.0, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            run_rolling_backtest(
                prices,
                transaction_cost_bps=invalid_cost,
            )


def test_rejects_boolean_transaction_cost(prices):
    import pytest

    with pytest.raises(ValueError):
        run_rolling_backtest(
            prices,
            transaction_cost_bps=True,
        )


@pytest.mark.parametrize(
    "covariance_method",
    ["sample", "shrinkage", "ledoit_wolf"],
)
def test_backtest_supports_all_covariance_estimators(
    prices,
    covariance_method,
):
    result = run_rolling_backtest(
        prices,
        estimation_window=100,
        rebalance_frequency=20,
        covariance_method=covariance_method,
    )

    assert not result.empty
    assert result.index.is_unique
    assert result.index.is_monotonic_increasing
    assert np.isfinite(result.to_numpy()).all()


def test_covariance_estimators_use_identical_out_of_sample_dates(prices):
    results = {
        method: run_rolling_backtest(
            prices,
            estimation_window=100,
            rebalance_frequency=20,
            covariance_method=method,
        )
        for method in ["sample", "shrinkage", "ledoit_wolf"]
    }

    reference_index = results["sample"].index

    for result in results.values():
        pd.testing.assert_index_equal(result.index, reference_index)


def test_default_covariance_method_is_sample(prices):
    default_result = run_rolling_backtest(
        prices,
        estimation_window=100,
        rebalance_frequency=20,
    )

    explicit_result = run_rolling_backtest(
        prices,
        estimation_window=100,
        rebalance_frequency=20,
        covariance_method="sample",
    )

    pd.testing.assert_frame_equal(default_result, explicit_result)


def test_rejects_unknown_covariance_method(prices):
    with pytest.raises(ValueError, match="Unknown covariance method"):
        run_rolling_backtest(
            prices,
            covariance_method="unknown",
        )
