
import pandas as pd


DEFAULT_SCENARIOS = {
    "Equity Sell-Off": {
        "US_EQUITY": -0.20,
        "EU_EQUITY": -0.20,
        "GOV_BONDS": 0.02,
        "CORPORATE_BONDS": -0.05,
        "GOLD": 0.05,
        "REITS": -0.15,
    },
    "Bond Sell-Off": {
        "US_EQUITY": -0.05,
        "EU_EQUITY": -0.05,
        "GOV_BONDS": -0.10,
        "CORPORATE_BONDS": -0.08,
        "GOLD": 0.02,
        "REITS": -0.05,
    },
    "Combined Risk-Off": {
        "US_EQUITY": -0.30,
        "EU_EQUITY": -0.30,
        "GOV_BONDS": 0.05,
        "CORPORATE_BONDS": -0.12,
        "GOLD": 0.10,
        "REITS": -0.25,
    },
}


def parametric_stress_report(
    portfolios: dict[str, pd.Series],
    scenarios: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Estimate portfolio returns under specified asset-level return shocks."""
    if not portfolios:
        raise ValueError("At least one portfolio is required.")

    scenarios = DEFAULT_SCENARIOS if scenarios is None else scenarios

    if not scenarios:
        raise ValueError("At least one scenario is required.")

    for portfolio_name, weights in portfolios.items():
        if not isinstance(weights, pd.Series):
            raise TypeError(f"Weights for '{portfolio_name}' must be a pandas Series.")

        if weights.empty or not weights.index.is_unique:
            raise ValueError(f"Weights for '{portfolio_name}' must have unique asset labels.")

        if not weights.index.isin(pd.Index(weights.index)).all():
            raise ValueError("Invalid asset labels.")

        if not pd.api.types.is_numeric_dtype(weights):
            raise TypeError(f"Weights for '{portfolio_name}' must be numeric.")

        if not weights.map(lambda x: pd.notna(x) and abs(x) != float("inf")).all():
            raise ValueError(f"Weights for '{portfolio_name}' must be finite.")

        if (weights < 0).any():
            raise ValueError(f"Weights for '{portfolio_name}' must be non-negative.")

        if abs(weights.sum() - 1.0) > 1e-8:
            raise ValueError(f"Weights for '{portfolio_name}' must sum to 1.")

    results = {}

    for scenario_name, shocks in scenarios.items():
        if not isinstance(shocks, dict) or not shocks:
            raise ValueError(f"Scenario '{scenario_name}' must contain asset shocks.")

        shock_series = pd.Series(shocks, dtype=float)

        if not shock_series.index.is_unique:
            raise ValueError(f"Scenario '{scenario_name}' has duplicate asset labels.")

        if not shock_series.map(lambda x: pd.notna(x) and abs(x) != float("inf")).all():
            raise ValueError(f"Scenario '{scenario_name}' shocks must be finite.")

        for portfolio_name, weights in portfolios.items():
            missing_assets = weights.index.difference(shock_series.index)
            if not missing_assets.empty:
                raise ValueError(
                    f"Scenario '{scenario_name}' is missing shocks for: "
                    f"{', '.join(missing_assets)}."
                )

            results.setdefault(portfolio_name, {})[scenario_name] = (
                weights * shock_series.reindex(weights.index)
            ).sum()

    report = pd.DataFrame(results).T
    report.index.name = "Portfolio"
    report.columns.name = "Scenario"
    return report
