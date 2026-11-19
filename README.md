# Portfolio Allocation & Risk-Based Optimization Engine

**Status:** In progress

A Python-based portfolio research project focused on portfolio construction, risk measurement, rolling out-of-sample backtesting, and quantitative evaluation of portfolio allocation strategies.

The project implements a minimum-variance portfolio workflow and compares it with an equal-weight benchmark. It also examines the effect of covariance estimation, portfolio rebalancing, transaction costs, and changing market conditions.

## Features

- Historical asset-return calculation and covariance estimation.
- Sample covariance, fixed shrinkage, and Ledoit-Wolf covariance estimators.
- Long-only minimum-variance portfolio optimization with configurable maximum asset weights.
- Equal-weight benchmark portfolio.
- Portfolio performance and risk analysis.
- Historical and parametric stress testing.
- Rolling out-of-sample backtesting with periodic rebalancing.
- Portfolio turnover and transaction-cost modelling.
- Backtest performance reporting.
- Covariance-estimator robustness and allocation-stability analysis.
- Fixed-weight and rolling portfolio performance attribution.
- Automated test suite.

## Project structure

```text
portfolio-allocation-engine/
├── app/
│   ├── compare_covariance.py
│   ├── compare_covariance_backtest.py
│   ├── covariance_robustness.py
│   ├── dashboard.py
│   ├── performance_attribution.py
│   ├── plot_analysis.py
│   ├── plot_backtest.py
│   ├── plot_sensitivity.py
│   ├── plot_weight_stability.py
│   ├── run_analysis.py
│   ├── run_backtest_report.py
│   ├── run_robustness.py
│   ├── run_sensitivity.py
│   ├── transaction_cost_sensitivity.py
│   └── weight_stability.py
├── backtesting/
│   ├── attribution.py
│   ├── rolling.py
│   └── weights.py
├── core/
│   ├── analysis.py
│   ├── attribution.py
│   ├── backtest_metrics.py
│   ├── backtest_reporting.py
│   ├── comparison.py
│   ├── covariance.py
│   ├── portfolio.py
│   ├── reporting.py
│   ├── returns.py
│   ├── risk.py
│   ├── robustness.py
│   ├── sensitivity.py
│   └── turnover.py
├── data/
│   ├── generate_data.py
│   └── prices.csv
├── optimization/
│   └── minimum_variance.py
├── scenarios/
│   ├── historical.py
│   └── parametric.py
├── tests/
├── LICENSE
├── README.md
├── pytest.ini
└── requirements.txt
```

## Methodology

### Asset returns and covariance

The project calculates asset returns from price data and estimates the covariance matrix used by the portfolio optimizer.

Three covariance approaches are available:

- **Sample covariance:** the standard historical covariance estimator.
- **Fixed shrinkage:** shrinks the sample covariance estimate towards a diagonal target.
- **Ledoit-Wolf:** estimates the shrinkage intensity from the data using the Ledoit-Wolf method.

The covariance estimator can materially affect optimized allocations, particularly when asset returns are noisy or correlated.

### Minimum-variance optimization

The optimizer seeks portfolio weights that minimize portfolio variance:

$$
\min_w \quad w^\mathsf{T}\Sigma w
$$

subject to:

$$
\sum_{i=1}^{N} w_i = 1,
\qquad
0 \leq w_i \leq w_{\max}
$$

where $w$ is the vector of portfolio weights, $\Sigma$ is the estimated covariance matrix, and $w_{max}$ is the maximum permitted weight for an individual asset.

The resulting portfolio is compared with an equal-weight allocation.

### Rolling out-of-sample backtesting

The rolling backtest uses a trailing estimation window to estimate covariance and determine portfolio weights. The portfolio is then held until the next scheduled rebalance.

The default configuration uses:

- Estimation window: 252 trading observations.
- Rebalancing frequency: 21 trading observations.
- Maximum individual asset weight: 40%.
- Transaction cost assumption: 10 basis points.

At each rebalance, the optimizer uses the trailing estimation window available at that point. Subsequent portfolio returns are evaluated out of sample until the next rebalance.

Portfolio weights drift with asset performance between rebalances. Executed turnover therefore accounts for the difference between the drifted holdings and the new target allocation.

Transaction costs are applied when the portfolio is rebalanced. If $TO_t$ is executed turnover and $b$ is the transaction-cost assumption in basis points, the proportional cost is:

$$
c_t = TO_t \frac{b}{10\,000}
$$

The net return on the rebalance day is calculated as:

$$
R_{t,\mathrm{net}} = (1+R_{t,\mathrm{gross}})(1-c_t)-1
$$

The backtest reports gross and net returns for both the minimum-variance and equal-weight strategies.

### Performance and risk evaluation

The backtest reporting workflow evaluates:

- Total return.
- Annualized return.
- Annualized volatility.
- Sharpe ratio.
- Maximum drawdown.

The Sharpe ratio uses the annualized arithmetic mean return in excess of the risk-free rate, divided by annualized volatility.

### Stress testing

The project includes historical and parametric stress analysis.

Historical stress testing evaluates portfolio behaviour over adverse historical return windows. Parametric scenarios apply specified shocks to asset classes to examine the sensitivity of portfolio allocations to adverse market conditions.

These scenarios are analytical tools, not forecasts of future market losses.

### Allocation stability and robustness

The project compares covariance estimators and examines how their resulting allocations change through time.

The analysis includes portfolio weight stability and allocation turnover across successive rebalances. This helps assess whether an optimizer's output is sensitive to the covariance estimation method.

### Performance attribution

Two attribution approaches are implemented:

- **Fixed-weight attribution:** decomposes portfolio performance into asset-level contributions for a fixed allocation.
- **Rolling attribution:** reconstructs daily beginning-of-period portfolio weights, including allocation drift and rebalancing, and decomposes rolling backtest returns into asset contributions and transaction costs.

The rolling attribution is designed to reconcile with the corresponding gross and net backtest returns.

## Synthetic data and reproducibility

The repository includes a deterministic synthetic price dataset generated by `data/generate_data.py`.

The generator simulates six asset classes using predefined annual return assumptions, volatilities, and cross-asset correlations. Daily multivariate Gaussian log returns are generated using a fixed random seed, then converted into price series.

The dataset is intended for testing and methodological demonstration. It is **not actual market data**, and results obtained from it should not be interpreted as evidence of live investment performance.

To regenerate the dataset:

```bash
python -m data.generate_data
```

## Example results

The following figures are illustrative results from the project's synthetic dataset and a particular full-history analysis configuration. They are not live or historical investment performance.

| Metric | Equal Weight | Minimum Variance |
|---|---:|---:|
| Annualized return | 5.92% | 4.47% |
| Annualized volatility | 9.07% | 5.55% |
| Sharpe ratio | 0.653 | 0.804 |
| Maximum drawdown | -17.86% | -10.72% |
| Worst 21-day return | -8.21% | -4.15% |

In this configuration, the minimum-variance portfolio had lower realized return and volatility than the equal-weight portfolio, alongside a higher Sharpe ratio and a smaller maximum drawdown.

Results depend on the dataset, estimation period, optimization constraints, covariance estimator, and backtest configuration. They should always be interpreted in that context.

## Installation

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
pip install -r requirements.txt
```

## Running the project

Run the portfolio analysis:

```bash
python app/run_analysis.py
```

Run the rolling backtest report:

```bash
python app/run_backtest_report.py
```

Launch the Streamlit dashboard:

```bash
streamlit run app/dashboard.py
```

Additional scripts in `app/` run covariance comparisons, robustness checks, sensitivity analysis, allocation-stability analysis, plotting, and performance attribution.

Generated reports and plots are written to the `reports/` directory where applicable. Generated report outputs are excluded from version control.

## Tests

Run the complete test suite from the repository root:

```bash
pytest -q
```

The tests cover portfolio calculations, covariance estimation, optimization, risk metrics, stress scenarios, rolling backtesting, reporting, turnover, and performance attribution.

## Limitations

- The included dataset is synthetic and does not represent actual market observations.
- The return-generating process is based on multivariate Gaussian log returns and does not capture all features of real financial markets, such as fat tails, volatility clustering, or changing correlations.
- Optimization results are sensitive to estimated inputs, especially the covariance matrix.
- The minimum-variance objective does not directly optimize expected return or investor-specific preferences.
- Transaction costs are represented by a simplified proportional basis-point assumption; market impact, liquidity constraints, taxes, and execution effects are not fully modelled.
- Backtest results depend on the chosen estimation window, rebalance frequency, constraints, and modelling assumptions.
- The project is a quantitative research and engineering exercise, not investment advice or a production trading system.

## Technologies

- Python
- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- Streamlit
- pytest
