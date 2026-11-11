import numpy as np
import pandas as pd
import pytest

from core.covariance import (
    calculate_covariance,
    calculate_shrinkage_covariance,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        rng.normal(
            loc=[0.0003, 0.0001, 0.0002],
            scale=[0.015, 0.004, 0.009],
            size=(500, 3),
        ),
        columns=["EQUITY", "BONDS", "GOLD"],
    )


def test_zero_shrinkage_equals_sample_covariance(returns):
    sample = calculate_covariance(returns, annualize=False)

    shrunk = calculate_shrinkage_covariance(
        returns,
        shrinkage=0.0,
        annualize=False,
    )

    pd.testing.assert_frame_equal(shrunk, sample)


def test_full_shrinkage_produces_diagonal_matrix(returns):
    result = calculate_shrinkage_covariance(
        returns,
        shrinkage=1.0,
        annualize=False,
    )

    expected = np.diag(np.diag(returns.cov().to_numpy()))

    np.testing.assert_allclose(result.to_numpy(), expected)


def test_partial_shrinkage_reduces_off_diagonal_covariances(returns):
    sample = calculate_covariance(returns, annualize=False)

    shrunk = calculate_shrinkage_covariance(
        returns,
        shrinkage=0.5,
        annualize=False,
    )

    sample_off_diagonal = sample.to_numpy() - np.diag(
        np.diag(sample.to_numpy())
    )

    shrunk_off_diagonal = shrunk.to_numpy() - np.diag(
        np.diag(shrunk.to_numpy())
    )

    np.testing.assert_allclose(
        shrunk_off_diagonal,
        0.5 * sample_off_diagonal,
    )


def test_shrinkage_preserves_asset_variances(returns):
    sample = calculate_covariance(returns, annualize=False)

    shrunk = calculate_shrinkage_covariance(
        returns,
        shrinkage=0.4,
        annualize=False,
    )

    np.testing.assert_allclose(
        np.diag(shrunk),
        np.diag(sample),
    )


def test_shrinkage_covariance_is_symmetric(returns):
    result = calculate_shrinkage_covariance(returns)

    np.testing.assert_allclose(
        result.to_numpy(),
        result.to_numpy().T,
    )


def test_shrinkage_covariance_is_annualized(returns):
    daily = calculate_shrinkage_covariance(
        returns,
        shrinkage=0.2,
        annualize=False,
    )

    annual = calculate_shrinkage_covariance(
        returns,
        shrinkage=0.2,
        annualize=True,
    )

    pd.testing.assert_frame_equal(annual, daily * 252)


@pytest.mark.parametrize("shrinkage", [-0.1, 1.1, np.inf, np.nan])
def test_invalid_shrinkage_is_rejected(returns, shrinkage):
    with pytest.raises(ValueError, match="Shrinkage"):
        calculate_shrinkage_covariance(
            returns,
            shrinkage=shrinkage,
        )


def test_missing_returns_are_rejected(returns):
    returns.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="missing"):
        calculate_shrinkage_covariance(returns)


def test_non_finite_returns_are_rejected(returns):
    returns.iloc[0, 0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        calculate_shrinkage_covariance(returns)


def test_duplicate_asset_labels_are_rejected(returns):
    returns.columns = ["EQUITY", "BONDS", "EQUITY"]

    with pytest.raises(ValueError, match="unique"):
        calculate_shrinkage_covariance(returns)


def test_insufficient_observations_are_rejected():
    returns = pd.DataFrame({"A": [0.01]})

    with pytest.raises(ValueError, match="two observations"):
        calculate_shrinkage_covariance(returns)
