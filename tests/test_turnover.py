import numpy as np
import pytest

from core.turnover import calculate_turnover


def test_identical_weights_have_zero_turnover():
    weights = np.array([0.5, 0.3, 0.2])

    assert calculate_turnover(weights, weights) == pytest.approx(0.0)


def test_turnover_is_half_sum_of_absolute_weight_changes():
    previous = np.array([0.5, 0.3, 0.2])
    target = np.array([0.3, 0.4, 0.3])

    assert calculate_turnover(previous, target) == pytest.approx(0.2)


def test_turnover_is_symmetric():
    previous = np.array([0.5, 0.3, 0.2])
    target = np.array([0.3, 0.4, 0.3])

    assert calculate_turnover(previous, target) == pytest.approx(
        calculate_turnover(target, previous)
    )


def test_rejects_different_array_shapes():
    with pytest.raises(ValueError, match="same shape"):
        calculate_turnover(
            np.array([0.5, 0.5]),
            np.array([0.3, 0.3, 0.4]),
        )


def test_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to one"):
        calculate_turnover(
            np.array([0.5, 0.4]),
            np.array([0.5, 0.5]),
        )


def test_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        calculate_turnover(
            np.array([1.1, -0.1]),
            np.array([0.5, 0.5]),
        )


def test_rejects_non_finite_weights():
    with pytest.raises(ValueError, match="finite"):
        calculate_turnover(
            np.array([np.nan, 1.0]),
            np.array([0.5, 0.5]),
        )
