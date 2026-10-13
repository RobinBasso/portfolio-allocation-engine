import numpy as np
import pandas as pd
import pytest

from optimization.minimum_variance import minimum_variance_weights


@pytest.fixture
def covariance():
    return pd.DataFrame(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.01],
            [0.00, 0.01, 0.16],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )


def test_weights_sum_to_one(covariance):
    weights = minimum_variance_weights(covariance)

    assert weights.sum() == pytest.approx(1.0)


def test_weights_respect_bounds(covariance):
    max_weight = 0.40

    weights = minimum_variance_weights(
        covariance,
        max_weight=max_weight,
    )

    assert (weights >= -1e-8).all()
    assert (weights <= max_weight + 1e-8).all()


def test_optimizer_reduces_variance_vs_equal_weight(covariance):
    weights = minimum_variance_weights(covariance)

    equal_weights = np.full(len(covariance), 1 / len(covariance))

    optimized_variance = (
        weights.to_numpy()
        @ covariance.to_numpy()
        @ weights.to_numpy()
    )

    equal_variance = (
        equal_weights
        @ covariance.to_numpy()
        @ equal_weights
    )

    assert optimized_variance <= equal_variance + 1e-10


def test_infeasible_weight_limit(covariance):
    with pytest.raises(ValueError):
        minimum_variance_weights(
            covariance,
            max_weight=0.30,
        )


def test_empty_covariance():
    with pytest.raises(ValueError):
        minimum_variance_weights(pd.DataFrame())


def test_asset_specific_weight_limits(covariance):
    limits = {"A": 0.20, "B": 0.50, "C": 0.60}

    weights = minimum_variance_weights(
        covariance,
        max_weight=limits,
    )

    assert weights.sum() == pytest.approx(1.0)
    for asset, limit in limits.items():
        assert weights[asset] <= limit + 1e-8


def test_missing_asset_limit_raises_error(covariance):
    with pytest.raises(ValueError, match="Missing weight limits"):
        minimum_variance_weights(
            covariance,
            max_weight={"A": 0.50, "B": 0.50},
        )


def test_unknown_asset_limit_raises_error(covariance):
    with pytest.raises(ValueError, match="Unknown assets"):
        minimum_variance_weights(
            covariance,
            max_weight={"A": 0.50, "B": 0.50, "C": 0.50, "D": 0.50},
        )


def test_limits_must_allow_full_investment(covariance):
    with pytest.raises(ValueError, match="too restrictive"):
        minimum_variance_weights(
            covariance,
            max_weight={"A": 0.20, "B": 0.30, "C": 0.40},
        )


def test_invalid_asset_limit_raises_error(covariance):
    with pytest.raises(ValueError, match="Weight limits must be in"):
        minimum_variance_weights(
            covariance,
            max_weight={"A": 0.20, "B": 0.50, "C": 1.10},
        )
