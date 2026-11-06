import numpy as np
import pandas as pd
import pytest

from app import plot_backtest


@pytest.fixture
def portfolio_returns():
    """Create deterministic returns for two portfolios."""
    dates = pd.bdate_range("2024-01-01", periods=100)

    return pd.DataFrame(
        {
            "Equal Weight": np.full(100, 0.001),
            "Minimum Variance": np.full(100, 0.0008),
        },
        index=dates,
    )


def test_plot_equity_curve_generates_chart(tmp_path, portfolio_returns, monkeypatch):
    monkeypatch.setattr(plot_backtest, "OUTPUT_DIR", tmp_path)

    plot_backtest.plot_equity_curve(portfolio_returns)

    assert (tmp_path / "backtest_equity_curve.png").is_file()


def test_plot_drawdowns_generates_chart(tmp_path, portfolio_returns, monkeypatch):
    monkeypatch.setattr(plot_backtest, "OUTPUT_DIR", tmp_path)

    plot_backtest.plot_drawdowns(portfolio_returns)

    assert (tmp_path / "backtest_drawdowns.png").is_file()


def test_main_generates_both_charts(tmp_path, monkeypatch):
    monkeypatch.setattr(plot_backtest, "OUTPUT_DIR", tmp_path)

    plot_backtest.main()

    assert (tmp_path / "backtest_equity_curve.png").is_file()
    assert (tmp_path / "backtest_drawdowns.png").is_file()
