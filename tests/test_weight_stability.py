
import numpy as np
import pandas as pd
import pytest

from app.weight_stability import (
    calculate_allocation_turnover,
    calculate_effective_number_of_assets,
    calculate_weight_statistics,
)


@pytest.fixture
def weight_history():
    return pd.DataFrame(
        {
            "US_EQUITY": [0.60, 0.40, 0.50],
            "GOV_BONDS": [0.40, 0.60, 0.50],
        },
        index=pd.to_datetime(
            ["2024-01-01", "2024-02-01", "2024-03-01"]
        ),
    )


def test_weight_statistics_returns_expected_columns(weight_history):
    statistics = calculate_weight_statistics(weight_history)

    assert list(statistics.columns) == [
        "Mean",
        "Std",
        "Minimum",
        "Maximum",
        "Range",
    ]

    assert list(statistics.index) == [
        "US_EQUITY",
        "GOV_BONDS",
    ]


def test_weight_statistics_calculates_mean_and_range(weight_history):
    statistics = calculate_weight_statistics(weight_history)

    assert statistics.loc["US_EQUITY", "Mean"] == pytest.approx(0.50)
    assert statistics.loc["US_EQUITY", "Minimum"] == pytest.approx(0.40)
    assert statistics.loc["US_EQUITY", "Maximum"] == pytest.approx(0.60)
    assert statistics.loc["US_EQUITY", "Range"] == pytest.approx(0.20)


def test_weight_statistics_calculates_sample_standard_deviation(
    weight_history,
):
    statistics = calculate_weight_statistics(weight_history)

    expected_std = np.std([0.60, 0.40, 0.50], ddof=1)

    assert statistics.loc["US_EQUITY", "Std"] == pytest.approx(
        expected_std
    )


def test_allocation_turnover_uses_half_sum_of_absolute_changes(
    weight_history,
):
    turnover = calculate_allocation_turnover(weight_history)

    expected = pd.Series(
        [0.20, 0.10],
        index=weight_history.index[1:],
    )

    pd.testing.assert_series_equal(
        turnover,
        expected,
        check_names=False,
    )


def test_allocation_turnover_excludes_first_observation(
    weight_history,
):
    turnover = calculate_allocation_turnover(weight_history)

    assert len(turnover) == len(weight_history) - 1
    assert turnover.index.equals(weight_history.index[1:])


def test_identical_allocations_have_zero_turnover():
    weights = pd.DataFrame(
        {
            "US_EQUITY": [0.60, 0.60, 0.60],
            "GOV_BONDS": [0.40, 0.40, 0.40],
        }
    )

    turnover = calculate_allocation_turnover(weights)

    assert (turnover == 0).all()


def test_effective_number_of_assets_is_one_for_single_asset():
    weights = pd.DataFrame({"US_EQUITY": [1.0]})

    effective_assets = calculate_effective_number_of_assets(weights)

    assert effective_assets.iloc[0] == pytest.approx(1.0)


def test_effective_number_of_assets_is_two_for_equal_weights():
    weights = pd.DataFrame(
        {
            "US_EQUITY": [0.50],
            "GOV_BONDS": [0.50],
        }
    )

    effective_assets = calculate_effective_number_of_assets(weights)

    assert effective_assets.iloc[0] == pytest.approx(2.0)


def test_effective_number_of_assets_uses_herfindahl_index(
    weight_history,
):
    effective_assets = calculate_effective_number_of_assets(
        weight_history
    )

    expected = 1 / weight_history.pow(2).sum(axis=1)

    pd.testing.assert_series_equal(
        effective_assets,
        expected,
    )


def test_effective_number_of_assets_increases_with_diversification():
    weights = pd.DataFrame(
        {
            "US_EQUITY": [0.80, 0.50],
            "GOV_BONDS": [0.20, 0.50],
        }
    )

    effective_assets = calculate_effective_number_of_assets(weights)

    assert effective_assets.iloc[1] > effective_assets.iloc[0]
