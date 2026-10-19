
# Portfolio Allocation & Risk-Based Optimization Engine

A Python portfolio analytics engine for asset allocation, risk attribution, constrained minimum-variance optimization, and portfolio stress testing.

The project demonstrates applied numerical methods, portfolio risk modeling, constrained optimization, software testing, and reproducible quantitative analysis.

## Features

- **Portfolio analytics:** annualized return, volatility, and Sharpe ratio.
- **Risk attribution:** marginal and component contributions to portfolio volatility.
- **Constrained optimization:** long-only minimum-variance allocation with full investment and asset-specific weight limits.
- **Historical stress testing:** maximum drawdown, worst rolling-period return, and worst daily return.
- **Parametric stress testing:** portfolio sensitivity to equity sell-offs, bond sell-offs, and combined risk-off scenarios.
- **Integrated reporting:** a single workflow combines performance, allocations, risk attribution, and stress results.
- **Reproducibility:** deterministic synthetic price data and automated tests.

## Project Structure

```text
portfolio-allocation-engine/
├── app/
│   └── run_analysis.py
├── core/
│   ├── analysis.py
│   ├── comparison.py
│   ├── covariance.py
│   ├── portfolio.py
│   ├── reporting.py
│   ├── returns.py
│   └── risk.py
├── optimization/
│   └── minimum_variance.py
├── scenarios/
│   ├── historical.py
│   └── parametric.py
├── tests/
├── data/
│   ├── generate_data.py
│   └── prices.csv
├── requirements.txt
└── README.md
```

## Methodology

### Returns and risk

Daily asset returns are calculated from the price series. Annualized covariance uses 252 trading days per year.

Portfolio volatility is derived from the covariance matrix and portfolio weights. Risk attribution decomposes portfolio volatility into asset-level contributions.

### Minimum-variance optimization

The optimizer minimizes portfolio variance subject to:

- Long-only asset weights.
- Full investment: weights sum to one.
- A maximum allocation constraint, either global or asset-specific.

The resulting portfolio minimizes estimated volatility under the supplied covariance matrix and constraints. It does not maximize expected return.

### Historical stress testing

Historical analysis measures realized portfolio drawdowns, worst daily returns, and worst rolling-period returns.

### Parametric stress testing

Hypothetical asset-level shocks are applied to portfolio weights to estimate scenario returns. The supplied scenarios are illustrative assumptions, not forecasts or probability estimates.

## Example Results

The following results are generated from the included synthetic dataset.

| Metric | Equal Weight | Minimum Variance |
|---|---:|---:|
| Annualized return | 5.92% | 4.47% |
| Annualized volatility | 9.07% | 5.55% |
| Sharpe ratio | 0.6529 | 0.8043 |
| Maximum drawdown | -17.86% | -10.72% |
| Worst 21-day return | -8.21% | -4.15% |

### Parametric Stress Results

| Scenario | Equal Weight | Minimum Variance |
|---|---:|---:|
| Equity sell-off | -8.83% | -3.46% |
| Bond sell-off | -5.17% | -7.55% |
| Combined risk-off | -13.67% | -5.91% |

The minimum-variance portfolio exhibits lower volatility and smaller losses under the equity sell-off and combined risk-off assumptions. It is more exposed to the modeled bond sell-off, reflecting its larger fixed-income allocation.

These results describe one synthetic dataset and a specific set of assumptions. They are not evidence of future performance.

## Installation

Requires Python 3.13 or a compatible Python version supported by the dependencies.

Clone the repository:

```bash
git clone git@github.com:RobinBasso/portfolio-allocation-engine.git
cd portfolio-allocation-engine
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Analysis

From the repository root:

```bash
python app/run_analysis.py
```

To run the test suite:

```bash
pytest
```

To regenerate the synthetic dataset:

```bash
python data/generate_data.py
```

## Limitations

- The included data are synthetic, not live market observations.
- Historical covariance and return estimates are sensitive to the chosen sample.
- Minimum-variance optimization can produce allocations that are concentrated in assets with low estimated volatility or favorable correlations.
- Parametric stress results depend on user-defined shocks and do not capture all market dynamics.
- Transaction costs, taxes, liquidity constraints, and market impact are not modeled.

## Technologies

Python, NumPy, pandas, SciPy, pytest, constrained numerical optimization, and quantitative risk analytics.
