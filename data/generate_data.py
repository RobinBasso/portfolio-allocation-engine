from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
TRADING_DAYS = 252 * 5

ASSETS = [
    "US_EQUITY",
    "EU_EQUITY",
    "GOV_BONDS",
    "CORPORATE_BONDS",
    "GOLD",
    "REITS",
]

ANNUAL_RETURNS = np.array([
    0.08,
    0.07,
    0.025,
    0.045,
    0.05,
    0.06,
])

ANNUAL_VOLATILITIES = np.array([
    0.18,
    0.19,
    0.06,
    0.08,
    0.16,
    0.20,
])

CORRELATION = np.array([
    [1.00, 0.75, -0.20, 0.10, 0.15, 0.60],
    [0.75, 1.00, -0.15, 0.10, 0.15, 0.55],
    [-0.20, -0.15, 1.00, 0.55, 0.05, -0.10],
    [0.10, 0.10, 0.55, 1.00, 0.10, 0.20],
    [0.15, 0.15, 0.05, 0.10, 1.00, 0.10],
    [0.60, 0.55, -0.10, 0.20, 0.10, 1.00],
])


def generate_prices() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    daily_returns = ANNUAL_RETURNS / 252
    daily_volatilities = ANNUAL_VOLATILITIES / np.sqrt(252)

    covariance = (
        np.outer(daily_volatilities, daily_volatilities)
        * CORRELATION
    )

    returns = rng.multivariate_normal(
        mean=daily_returns,
        cov=covariance,
        size=TRADING_DAYS,
    )

    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    dates = pd.bdate_range(
        end="2025-12-31",
        periods=TRADING_DAYS,
    )

    return pd.DataFrame(
        prices,
        index=dates,
        columns=ASSETS,
    )


if __name__ == "__main__":
    output_path = Path(__file__).parent / "prices.csv"

    prices = generate_prices()
    prices.to_csv(output_path)

    print(f"Saved {len(prices)} observations to {output_path}")