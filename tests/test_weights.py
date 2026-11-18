import numpy as np
import pandas as pd
import pytest

from backtesting.weights import collect_rolling_weights


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

    return pd.DataFrame(
        values,
        index=dates,
        columns=assets,
    )


def test_returns_expected_number_of_rebalances(prices):
    result = collect_rolling_weights(
        prices,
        estimation_window=100,
        rebalance_frequency=20,
    )

    expected_count = len(range(100, len(prices), 20))

    assert len(result) == expected_count


def test_first_weights_use_only_training_data(prices):
    result = collect_rolling_weights(
        prices,
        estimation_window=100,
        rebalance_frequency=20,
    )

    assert result.index[0] == prices.index[100]


def test_weight_dates_are_unique_and_ascending(prices):
    result = collect_rolling_weights(prices)

    assert result.index.is_unique
    assert result.index.is_monotonic_increasing


def test_weight_columns_match_asset_columns(prices):
    result = collect_rolling_weights(prices)

    assert list(result.columns) == list(prices.columns)


def test_weights_sum_to_one(prices):
    result = collect_rolling_weights(prices)

    np.testing.assert_allclose(
        result.sum(axis=1).to_numpy(),
        1.0,
        atol=1e-8,
    )


def test_weights_respect_long_only_constraint(prices):
    result = collect_rolling_weights(prices)

    assert (result >= -1e-10).all().all()


def test_weights_respect_maximum_weight(prices):
    result = collect_rolling_weights(
        prices,
        max_weight=0.40,
    )

    assert (result <= 0.40 + 1e-8).all().all()


@pytest.mark.parametrize(
    "method",
    ["sample", "shrinkage", "ledoit_wolf"],
)
def test_supports_all_covariance_estimators(prices, method):
    result = collect_rolling_weights(
        prices,
        estimation_window=100,
        rebalance_frequency=20,
        covariance_method=method,
    )

    assert not result.empty
    assert np.isfinite(result.to_numpy()).all()


def test_rejects_unknown_covariance_estimator(prices):
    with pytest.raises(ValueError, match="Unknown covariance method"):
        collect_rolling_weights(
            prices,
            covariance_method="unknown",
        )


def test_rejects_insufficient_price_history(prices):
    with pytest.raises(ValueError, match="insufficient"):
        collect_rolling_weights(
            prices.iloc[:100],
            estimation_window=100,
        )


def test_rejects_duplicate_asset_columns(prices):
    prices.columns = ["A", "B", "A"]

    with pytest.raises(ValueError, match="unique"):
        collect_rolling_weights(prices)


def test_rejects_unsorted_dates(prices):
    prices = prices.iloc[::-1]

    with pytest.raises(ValueError, match="ascending"):
        collect_rolling_weights(prices)


def test_rejects_invalid_estimation_window(prices):
    with pytest.raises(ValueError, match="at least 2"):
        collect_rolling_weights(
            prices,
            estimation_window=1,
        )


def test_rejects_invalid_rebalance_frequency(prices):
    with pytest.raises(ValueError, match="at least 1"):
        collect_rolling_weights(
            prices,
            rebalance_frequency=0,
        )