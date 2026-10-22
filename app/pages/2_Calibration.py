import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

from common import load_analysis, load_matches, page_setup, season_filter
from src import calibration

page_setup("Market Calibration", icon="🎚️")
st.caption(
    "When the market says 60%, does it happen 60% of the time? "
    "This reliability curve is the platform's signature chart."
)

df = season_filter(load_matches())
prob_source = st.sidebar.selectbox(
    "Probability source",
    ["p_cons", "p_PS", "p_B365", "p_PSC", "p_Avg", "p_Max"],
    format_func=lambda c: {
        "p_cons": "Market consensus",
        "p_PS": "Pinnacle",
        "p_B365": "Bet365",
        "p_PSC": "Pinnacle closing",
        "p_Avg": "Market average",
        "p_Max": "Best price",
    }[c],
)

rel = calibration.reliability_curve(df, prob_source)
if rel.empty:
    st.warning("No matches with odds from this source in the selected seasons.")
    st.stop()

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
               line=dict(dash="dash", color="gray"))
)
fig.add_trace(
    go.Scatter(
        x=rel["predicted"],
        y=rel["observed"],
        error_y=dict(
            type="data",
            symmetric=False,
            array=rel["ci_hi"] - rel["observed"],
            arrayminus=rel["observed"] - rel["ci_lo"],
        ),
        mode="markers+lines",
        name="observed frequency",
        marker=dict(size=9),
        customdata=rel["n"],
        hovertemplate="predicted %{x:.1%}<br>observed %{y:.1%}<br>n=%{customdata}",
    )
)
fig.update_layout(
    xaxis=dict(title="market implied probability (de-margined)", tickformat=".0%"),
    yaxis=dict(title="observed outcome frequency", tickformat=".0%"),
    height=560,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Efficiency regression")
st.markdown(
    "Logistic regression of outcomes on the logit of implied probability. "
    "An efficient market has **slope = 1** and **intercept = 0**; a slope "
    "above 1 means favorites are underpriced at the extremes "
    "(favorite–longshot bias)."
)
try:
    eff = calibration.efficiency_regression(df, prob_source)
    c1, c2, c3 = st.columns(3)
    c1.metric("Slope", f"{eff['slope']:.3f} ± {eff['slope_se']:.3f}")
    c2.metric("Intercept", f"{eff['intercept']:.3f} ± {eff['intercept_se']:.3f}")
    c3.metric("Observations", f"{eff['n']:,}")
    st.caption(
        f"Wald z vs efficiency: slope z = {eff['z_slope_vs_1']:+.2f}, "
        f"intercept z = {eff['z_intercept_vs_0']:+.2f} "
        "(|z| > 2 indicates a statistically detectable deviation — with this "
        "much data, statistically detectable is not the same as exploitable)."
    )
except Exception:
    st.warning("Not enough data for the efficiency regression on this selection.")

full = load_analysis()
periods = full.get("efficiency_by_period")
if periods:
    st.subheader("Early vs late season (all seasons)")
    st.markdown(
        "Calibration re-estimated separately for August–December and "
        "January–May: does the market sharpen as team information accumulates?"
    )
    cols = st.columns(len(periods))
    for col, row in zip(cols, periods):
        col.metric(
            f"{row['period']} season slope",
            f"{row['slope']:.3f} ± {row['slope_se']:.3f}",
        )
