import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.analysis import run_portfolio_analysis


st.set_page_config(
    page_title="Portfolio Allocation & Risk Engine",
    page_icon="",
    layout="wide",
)

st.title("Portfolio Allocation & Risk Engine")
st.caption("Portfolio construction, risk attribution and stress analysis")

st.sidebar.header("Analysis parameters")

risk_free_rate = st.sidebar.number_input(
    "Annual risk-free rate",
    min_value=-0.05,
    max_value=0.20,
    value=0.0,
    step=0.005,
    format="%.3f",
)

stress_window = st.sidebar.number_input(
    "Historical stress window (days)",
    min_value=1,
    max_value=252,
    value=21,
    step=1,
)

prices = pd.read_csv(
    PROJECT_ROOT / "data" / "prices.csv",
    index_col=0,
    parse_dates=True,
)

analysis = run_portfolio_analysis(
    prices,
    risk_free_rate=risk_free_rate,
    stress_window=int(stress_window),
)

st.header("Portfolio performance")

comparison = analysis["comparison"]

columns = st.columns(len(comparison))

for column, (portfolio, metrics) in zip(columns, comparison.iterrows()):
    with column:
        st.subheader(portfolio)
        st.metric("Annualized return", f"{metrics['Annualized Return']:.2%}")
        st.metric("Annualized volatility", f"{metrics['Annualized Volatility']:.2%}")
        st.metric("Sharpe ratio", f"{metrics['Sharpe Ratio']:.3f}")

st.header("Asset allocation")

allocations = pd.DataFrame(analysis["allocations"])
st.bar_chart(allocations)

st.dataframe(
    allocations.style.format("{:.2%}"),
    use_container_width=True,
)

st.header("Risk contribution")

risk_contributions = pd.DataFrame({
    portfolio: report["asset_report"]["Risk Contribution"]
    for portfolio, report in analysis["risk_reports"].items()
})

st.bar_chart(risk_contributions)

st.dataframe(
    risk_contributions.style.format("{:.2%}"),
    use_container_width=True,
)

st.header("Historical stress analysis")

historical_stress = analysis["historical_stress"]

st.dataframe(
    historical_stress.style.format("{:.2%}"),
    use_container_width=True,
)

st.header("Parametric stress analysis")

parametric_stress = analysis["parametric_stress"]

st.dataframe(
    parametric_stress.style.format("{:.2%}"),
    use_container_width=True,
)