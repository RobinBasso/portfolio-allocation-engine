
# Portfolio Allocation & Risk-Based Optimization Engine

A Python portfolio analytics and optimization engine for constrained asset allocation, portfolio risk measurement, out-of-sample backtesting, and quantitative robustness analysis.

The project demonstrates applied numerical methods, covariance estimation, portfolio optimization, risk attribution, transaction cost modeling, and systematic testing.

## Project Status

**Status: In Progress**

The engine currently includes:

- Constrained minimum-variance portfolio optimization.
- Historical and parametric stress testing.
- Portfolio risk attribution.
- Rolling out-of-sample backtesting.
- Transaction cost sensitivity analysis.
- Covariance estimator robustness analysis.
- Portfolio allocation stability analysis.
- Rolling portfolio performance attribution, including daily asset contributions and transaction costs.
- Automated unit and integration tests.

## Features

- **Portfolio analytics:** annualized return, volatility, and Sharpe ratio.
- **Risk attribution:** marginal and component contributions to portfolio volatility.
- **Constrained optimization:** long-only minimum-variance allocation with full investment and asset-specific weight limits.
- **Covariance estimation:** sample covariance, fixed shrinkage, and Ledoit-Wolf estimators.
- **Rolling backtesting:** periodic portfolio re-optimization using historical estimation windows.
- **Transaction cost modeling:** gross and net performance analysis under different transaction cost assumptions.
- **Historical stress testing:** maximum drawdown, worst rolling-period return, and worst daily return.
- **Parametric stress testing:** portfolio sensitivity to equity sell-offs, bond sell-offs, and combined risk-off scenarios.
- **Robustness analysis:** comparison of covariance estimators, estimation windows, and rebalancing configurations.
- **Allocation stability:** analysis of weight variability, target allocation turnover, and effective portfolio diversification.
- **Reproducibility:** deterministic synthetic price data and automated tests.

## Project Structure

```text
portfolio-allocation-engine/
├── app/
│   ├── covariance_robustness.py
│   ├── performance_attribution.py
│   ├── plot_analysis.py
│   ├── plot_backtest.py
│   ├── plot_weight_stability.py
│   ├── run_analysis.py
│   ├── transaction_cost_sensitivity.py
│   └── weight_stability.py
│
├── backtesting/
│   ├── attribution.py
│   ├── rolling.py
│   └── weights.py
├── core/
│   ├── analysis.py
│   ├── attribution.py
│   ├── backtest_metrics.py
│   ├── comparison.py
│   ├── covariance.py
│   ├── portfolio.py
│   ├── reporting.py
│   ├── returns.py
│   ├── risk.py
│   └── turnover.py
│
├── optimization/
│   └── minimum_variance.py
├── scenarios/
│   ├── historical.py
│   └── parametric.py
├── tests/
├── data/
│   ├── generate_data.py
│   └── prices.csv
├── reports/
├── requirements.txt
└── README.md
```

## Methodology

### Returns and risk

Daily asset returns are calculated from historical price series. Annualized covariance uses 252 trading days per year.

Portfolio volatility is derived from the covariance matrix and portfolio weights. Risk attribution decomposes portfolio volatility into asset-level contributions.

### Covariance estimation

Three covariance estimators are implemented and compared:

- **Sample covariance:** estimates covariance directly from historical returns.
- **Fixed shrinkage:** shrinks sample covariance estimates towards a diagonal covariance target.
- **Ledoit-Wolf:** applies a data-driven shrinkage estimator.

The objective is to assess how covariance estimation affects portfolio construction, realized performance, and allocation stability.

### Minimum-variance optimization

The optimizer minimizes portfolio variance subject to:

- Long-only asset weights.
- Full investment: weights sum to one.
- A maximum allocation constraint, either global or asset-specific.

The resulting portfolio minimizes estimated volatility under the supplied covariance matrix and constraints. It does not maximize expected return.

### Rolling out-of-sample backtesting

The backtesting engine uses a rolling estimation window to estimate covariance and periodically re-optimize portfolio weights.

The default configuration uses:

- Estimation window: 252 observations.
- Rebalancing frequency: 21 observations.
- Maximum asset allocation: 40%.

Portfolio weights are allowed to drift between rebalancing dates as asset returns change their relative portfolio values.

The backtest tracks portfolio returns and executed turnover, allowing performance to be evaluated before and after transaction costs.

### Transaction cost sensitivity

Transaction costs are applied to executed portfolio turnover at rebalancing dates.

The analysis compares gross and net performance under different transaction cost assumptions. Configurations are evaluated over a common out-of-sample period to improve comparability.

### Covariance estimator robustness

The robustness analysis compares portfolio performance across covariance estimators and alternative backtesting configurations.

The purpose is to assess whether portfolio results are sensitive to the covariance estimation method and model parameters.

### Allocation stability

The allocation stability analysis examines the weights generated at each rebalancing date.

Three complementary measures are considered:

- **Weight variability:** standard deviation and range of each asset's target allocation.
- **Target allocation turnover:** one-way turnover between consecutive optimized target allocations, calculated as half the sum of absolute weight changes.
- **Effective number of assets:** inverse Herfindahl index, calculated as \(1 / \sum_i w_i^2\).

Target allocation turnover measures changes between optimized allocations. It is distinct from executed trading turnover, which also accounts for portfolio weight drift.

### Rolling portfolio performance attribution

Rolling performance attribution decomposes daily portfolio returns into asset-level contributions using beginning-of-day portfolio weights.

For each asset, the daily contribution is calculated as:

\[
C_{i,t} = w_{i,t-1} r_{i,t}
\]

where \(w_{i,t-1}\) represents the asset's portfolio weight at the beginning of the day and \(r_{i,t}\) its daily return.

Portfolio weights drift between rebalancing dates as asset prices evolve. At each rebalance, weights are reset to the newly optimized target allocation.

Daily asset contributions reconcile to the gross portfolio return:

\[
R_{p,t}^{gross} = \sum_i C_{i,t}
\]

Transaction costs are calculated separately from executed turnover and deducted from gross returns:

\[
R_{p,t}^{net} = R_{p,t}^{gross} - TC_t
\]

This separates investment performance attributable to asset exposures from the impact of trading costs.

The attribution implementation is validated against the rolling backtest, ensuring that gross and net daily returns reconcile for both Equal Weight and Minimum Variance strategies.

## Example Results

The following portfolio results are generated from the included synthetic dataset.

| Metric | Equal Weight | Minimum Variance |
|---|---:|---:|
| Annualized return | 5.92% | 4.47% |
| Annualized volatility | 9.07% | 5.55% |
| Sharpe ratio | 0.6529 | 0.8043 |
| Maximum drawdown | -17.86% | -10.72% |
| Worst 21-day return | -8.21% | -4.15% |

### Parametric stress results

| Scenario | Equal Weight | Minimum Variance |
|---|---:|---:|
| Equity sell-off | -8.83% | -3.46% |
| Bond sell-off | -5.17% | -7.55% |
| Combined risk-off | -13.67% | -5.91% |

The minimum-variance portfolio exhibits lower volatility and smaller losses under the equity sell-off and combined risk-off assumptions. It is more exposed to the modeled bond sell-off, reflecting its larger fixed-income allocation.

### Allocation stability results

The following results compare covariance estimators over 36 rebalancing dates, from February 2023 to December 2025.

| Metric | Sample | Fixed Shrinkage | Ledoit-Wolf |
|---|---:|---:|---:|
| Mean target allocation turnover | 2.06% | 1.48% | 1.88% |
| Median target allocation turnover | 1.90% | 1.54% | 1.85% |
| Maximum target allocation turnover | 4.55% | 2.64% | 3.15% |
| Mean effective number of assets | 3.229 | 3.289 | 3.294 |

Fixed shrinkage produces the lowest target allocation turnover in this experiment. It also reduces the variability of equity allocations compared with sample covariance.

Government bonds remain at the 40% maximum allocation throughout the analysis. This reflects the binding optimization constraint rather than an unconstrained preference of the covariance estimator.

These results suggest that covariance shrinkage can improve allocation stability in this particular configuration. They do not establish that one estimator will perform best across other datasets or market regimes.

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

## Running the Analysis

Run the integrated portfolio analysis:

```bash
python -m app.run_analysis
```

Run covariance estimator robustness analysis:

```bash
python -m app.covariance_robustness
```

Run transaction cost sensitivity analysis:

```bash
python -m app.transaction_cost_sensitivity
```

Run allocation stability analysis:

```bash
python -m app.weight_stability
```

Run rolling portfolio performance attribution:

```bash
python -m app.performance_attribution
```

Generate allocation stability charts:

```bash
python -m app.plot_weight_stability
```

Other portfolio charts can be generated with:

```bash
python -m app.plot_analysis
python -m app.plot_backtest
```

Generated charts are saved in the `reports/` directory.

## Testing

Run the complete test suite:

```bash
pytest -v
```

The project includes tests covering portfolio analytics, covariance estimation, optimization constraints, rolling backtesting, transaction costs, allocation stability, and reporting.

## Limitations

- The included data are synthetic, not live market observations.
- Historical covariance and return estimates are sensitive to the chosen sample.
- Minimum-variance optimization can produce allocations concentrated in assets with low estimated volatility or favorable correlations.
- Backtest results depend on estimation windows, rebalancing frequency, and covariance model assumptions.
- Transaction costs are simplified and do not fully capture liquidity constraints or market impact.
- Parametric stress results depend on user-defined shocks and do not capture all market dynamics.
- Results from synthetic data should not be interpreted as evidence of future investment performance.

## Technologies

Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib, pytest, constrained numerical optimization, covariance estimation, and quantitative portfolio risk analytics.
