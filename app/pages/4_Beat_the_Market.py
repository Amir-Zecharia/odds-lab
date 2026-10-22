import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import league_selector, load_analysis, page_setup

page_setup("Beat the Market?", icon="⚔️")
st.caption(
    "The platform's central question: does anything add predictive value "
    "beyond the closing-odds consensus? Expected answer: no. "
    "Any exception is a finding."
)

league_selector()
analysis = load_analysis()

st.subheader("Out-of-sample comparison")
st.markdown(
    """
Every model below sees the market consensus; the question is whether extra
features (cross-book dispersion, sharp–soft gap, Elo, form, rest days) improve
**out-of-sample log loss**, evaluated with an expanding window: train on
seasons 1..k, predict season k+1, repeat. XGBoost repeats the exercise
without linearity assumptions.
"""
)

oos = pd.DataFrame(analysis["oos_comparison"])
market_ll = oos.loc[oos["model"] == "market consensus (no fit)", "log_loss"].iloc[0]

fig = go.Figure(
    go.Scatter(
        x=oos["diff_vs_market"],
        y=oos["model"],
        error_x=dict(
            type="data",
            symmetric=False,
            array=oos["ci_hi"] - oos["diff_vs_market"],
            arrayminus=oos["diff_vs_market"] - oos["ci_lo"],
        ),
        mode="markers",
        marker=dict(size=11),
    )
)
fig.add_vline(x=0, line_dash="dash", line_color="gray")
fig.update_layout(
    xaxis_title=(
        "log-loss difference vs raw market consensus "
        "(negative = beats the market) with 95% paired-bootstrap CI"
    ),
    height=90 + 55 * len(oos),
)
st.plotly_chart(fig, use_container_width=True)

show = oos.rename(
    columns={
        "log_loss": "log loss",
        "diff_vs_market": "Δ vs market",
        "ci_lo": "CI low",
        "ci_hi": "CI high",
    }
).set_index("model")
st.dataframe(
    show[["log loss", "Δ vs market", "CI low", "CI high", "n_matches"]]
    .style.format({"log loss": "{:.4f}", "Δ vs market": "{:+.4f}",
                   "CI low": "{:+.4f}", "CI high": "{:+.4f}"}),
    use_container_width=True,
)
st.caption(
    f"Raw market consensus: {market_ll:.4f} log loss over "
    f"{int(oos['n_matches'].iloc[0]):,} test matches. A model beats the "
    "market only if its CI lies entirely below zero."
)

st.subheader("In-sample likelihood-ratio tests")
st.markdown(
    "Does each feature set improve the fit *at all*, before asking whether "
    "it generalizes? Nested multinomial logit vs the consensus-only baseline."
)
lrt = pd.DataFrame(analysis["lr_tests"]).set_index("model")
lrt["significant (5%)"] = lrt["p_value"] < 0.05
st.dataframe(
    lrt[["lr_stat", "df", "p_value", "significant (5%)"]]
    .style.format({"lr_stat": "{:.2f}", "p_value": "{:.4f}"}),
    use_container_width=True,
)
st.caption(
    f"Fitted on {analysis['model_rows']:,} matches with complete features "
    "(Pinnacle era, 2012-13 onward). In-sample significance without "
    "out-of-sample improvement is the classic signature of overfitting "
    "an efficient market."
)
