import numpy as np
import pandas as pd
import pytest

from core.covariance import (
    calculate_ledoit_wolf_covariance,
    TRADING_DAYS_PER_YEAR,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)

    values = rng.normal(
        loc=0.0003,
        scale=0.01,
        size=(500, 4),
    )

    return pd.DataFrame(
        values,
        columns=["US_EQUITY", "EU_EQUITY", "BONDS", "GOLD"],
    )


def test_returns_dataframe_with_original_asset_labels(returns):
    covariance = calculate_ledoit_wolf_covariance(returns)

    assert isinstance(covariance, pd.DataFrame)
    assert list(covariance.index) == list(returns.columns)
    assert list(covariance.columns) == list(returns.columns)


def test_covariance_matrix_is_symmetric(returns):
    covariance = calculate_ledoit_wolf_covariance(returns)

    np.testing.assert_allclose(
        covariance.to_numpy(),
        covariance.to_numpy().T,
    )


def test_covariance_matrix_is_positive_semidefinite(returns):
    covariance = calculate_ledoit_wolf_covariance(returns)

    eigenvalues = np.linalg.eigvalsh(covariance.to_numpy())

    assert np.all(eigenvalues >= -1e-10)


def test_annualization(returns):
    daily = calculate_ledoit_wolf_covariance(
        returns,
        annualize=False,
    )

    annualized = calculate_ledoit_wolf_covariance(
        returns,
        annualize=True,
    )

    np.testing.assert_allclose(
        annualized.to_numpy(),
        daily.to_numpy() * TRADING_DAYS_PER_YEAR,
    )


def test_no_annualization(returns):
    covariance = calculate_ledoit_wolf_covariance(
        returns,
        annualize=False,
    )

    assert covariance.to_numpy().max() < 1.0


def test_empty_returns_rejected():
    with pytest.raises(ValueError):
        calculate_ledoit_wolf_covariance(pd.DataFrame())


def test_missing_values_rejected():
    returns = pd.DataFrame({
        "A": [0.01, 0.02, np.nan],
        "B": [0.03, 0.01, 0.02],
    })

    with pytest.raises(ValueError):
        calculate_ledoit_wolf_covariance(returns)


def test_non_finite_values_rejected():
    returns = pd.DataFrame({
        "A": [0.01, 0.02, np.inf],
        "B": [0.03, 0.01, 0.02],
    })

    with pytest.raises(ValueError):
        calculate_ledoit_wolf_covariance(returns)


def test_duplicate_asset_labels_rejected():
    returns = pd.DataFrame(
        [[0.01, 0.02], [0.03, 0.01]],
        columns=["A", "A"],
    )

    with pytest.raises(ValueError):
        calculate_ledoit_wolf_covariance(returns)


def test_insufficient_observations_rejected():
    returns = pd.DataFrame({
        "A": [0.01],
        "B": [0.02],
    })

    with pytest.raises(ValueError):
        calculate_ledoit_wolf_covariance(returns)