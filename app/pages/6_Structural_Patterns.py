import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import league_filter, load_analysis, load_matches, page_setup
from src import calibration

page_setup("Structural Patterns", icon="🏟️")

df = league_filter(load_matches())
analysis = load_analysis()

st.subheader("Home advantage: the empty-stadium natural experiment")
st.markdown(
    "Realized home-win rate vs the market's expected home probability. "
    "In **2020-21** matches were played in empty stadiums — home advantage "
    "collapsed, and the gap shows how quickly the market adapted."
)
ha = calibration.home_advantage_by_season(df).reset_index()
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=ha["season"], y=ha["home_win_rate"], name="realized home-win rate",
               mode="lines+markers")
)
fig.add_trace(
    go.Scatter(x=ha["season"], y=ha["market_home_prob"], name="market expectation",
               mode="lines+markers", line=dict(dash="dot"))
)
# Categorical x-axis: address the season by its category index.
if "2020-21" in list(ha["season"]):
    covid = list(ha["season"]).index("2020-21")
    fig.add_vrect(
        x0=covid - 0.5, x1=covid + 0.5, fillcolor="red", opacity=0.15,
        line_width=0, annotation_text="empty stadiums",
        annotation_position="top left",
    )
fig.update_yaxes(tickformat=".0%", title="home-win probability")
fig.update_layout(height=480)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Draw pricing")
dp = calibration.draw_pricing_by_season(df).reset_index()
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=dp["season"], y=dp["realized_draw_rate"], name="realized draw rate",
               mode="lines+markers")
)
fig.add_trace(
    go.Scatter(x=dp["season"], y=dp["market_draw_prob"], name="market draw probability",
               mode="lines+markers", line=dict(dash="dot"))
)
fig.update_yaxes(tickformat=".0%")
fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Dixon-Coles: how far does a stats-only model trail the market?")
st.markdown(
    "A Poisson goal model (Dixon & Coles 1997: team attack/defence strengths, "
    "home advantage, low-score correction, time-decayed fitting) built "
    "**without any odds information**, refitted before each test season."
)
dc = analysis["dixon_coles"]
table = pd.DataFrame(dc["summary"]).set_index("model")
st.dataframe(table.style.format({"log_loss": "{:.4f}"}), use_container_width=True)
st.markdown(
    f"Gap: **{dc['diff_vs_market']:+.4f} nats/match** "
    f"(95% CI [{dc['ci_lo']:+.4f}, {dc['ci_hi']:+.4f}]). "
    "This is the price of ignoring the market: a competent statistical model "
    "gets close, but the closing line knows more — injuries, lineups, "
    "weather, and the aggregated opinions of every informed bettor."
)
