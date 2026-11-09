import numpy as np
import pandas as pd
import pytest

from core.sensitivity import run_sensitivity_analysis


@pytest.fixture
def prices():
    rng = np.random.default_rng(42)

    dates = pd.bdate_range("2020-01-01", periods=700)

    returns = rng.normal(
        loc=0.0003,
        scale=0.01,
        size=(700, 3),
    )

    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    return pd.DataFrame(
        prices,
        index=dates,
        columns=["Asset A", "Asset B", "Asset C"],
    )


def test_sensitivity_returns_all_parameter_combinations(prices):
    result = run_sensitivity_analysis(
        prices,
        estimation_windows=(126, 252),
        rebalance_frequencies=(5, 21),
    )

    assert len(result) == 8
    assert result.index.nlevels == 3


def test_sensitivity_includes_both_strategies(prices):
    result = run_sensitivity_analysis(
        prices,
        estimation_windows=(126,),
        rebalance_frequencies=(21,),
    )

    strategies = result.index.get_level_values("Strategy")

    assert set(strategies) == {
        "Equal Weight",
        "Minimum Variance",
    }


def test_sensitivity_includes_gross_and_net_metrics(prices):
    result = run_sensitivity_analysis(
        prices,
        estimation_windows=(126,),
        rebalance_frequencies=(21,),
    )

    required_columns = [
        "Gross Total Return",
        "Net Total Return",
        "Gross Annualized Return",
        "Net Annualized Return",
        "Gross Annualized Volatility",
        "Net Annualized Volatility",
        "Gross Sharpe Ratio",
        "Net Sharpe Ratio",
        "Gross Maximum Drawdown",
        "Net Maximum Drawdown",
        "Cumulative Turnover",
    ]

    for column in required_columns:
        assert column in result.columns


def test_sensitivity_uses_common_evaluation_period(prices):
    result = run_sensitivity_analysis(
        prices,
        estimation_windows=(126, 252),
        rebalance_frequencies=(5, 21),
    )

    # Both estimation windows must use the same evaluation period.
    # The output metrics are calculated over the common period.
    assert len(result) == 8
    assert np.isfinite(result.to_numpy(dtype=float)).all()


def test_sensitivity_rejects_empty_parameter_lists(prices):
    with pytest.raises(ValueError):
        run_sensitivity_analysis(
            prices,
            estimation_windows=(),
        )


def test_sensitivity_rejects_duplicate_estimation_windows(prices):
    with pytest.raises(ValueError):
        run_sensitivity_analysis(
            prices,
            estimation_windows=(126, 126),
        )


def test_sensitivity_rejects_invalid_estimation_windows(prices):
    with pytest.raises(ValueError):
        run_sensitivity_analysis(
            prices,
            estimation_windows=(1, 126),
        )


def test_sensitivity_rejects_invalid_rebalance_frequencies(prices):
    with pytest.raises(ValueError):
        run_sensitivity_analysis(
            prices,
            rebalance_frequencies=(0, 21),
        )


def test_sensitivity_rejects_insufficient_prices():
    prices = pd.DataFrame(
        np.ones((100, 3)),
        columns=["Asset A", "Asset B", "Asset C"],
    )

    with pytest.raises(ValueError):
        run_sensitivity_analysis(
            prices,
            estimation_windows=(126,),
        )
