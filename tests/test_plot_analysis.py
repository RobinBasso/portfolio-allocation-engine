from app import plot_analysis


def test_plot_analysis_generates_both_charts(tmp_path, monkeypatch):
    """Generate both charts in an isolated temporary directory."""
    monkeypatch.setattr(plot_analysis, "OUTPUT_DIR", tmp_path)

    plot_analysis.main()

    assert (tmp_path / "portfolio_allocations.png").is_file()
    assert (tmp_path / "risk_attribution.png").is_file()
