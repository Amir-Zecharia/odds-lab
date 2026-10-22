import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import load_matches, page_setup, season_filter
from src.evaluate import bootstrap_ci
from src.totals import (
    TOTAL_BOOK_NAMES,
    available_total_books,
    binary_log_loss,
    efficiency_binary,
    reliability_binary,
)

page_setup("Totals Market", icon="🥅")
st.caption(
    "A second, independent market: over/under 2.5 goals. Binary outcomes, "
    "same machinery — is the totals market as efficient as match odds?"
)

df = season_filter(load_matches())
books = [b for b in available_total_books(df) if df[f"p_over_{b}"].notna().any()]
if not books:
    st.warning("No totals odds in the selected seasons (coverage starts 2005-06).")
    st.stop()

covered = df.dropna(subset=["p_over_Avg"]) if "p_over_Avg" in df.columns else df
c1, c2, c3 = st.columns(3)
c1.metric("Matches with totals odds", f"{len(covered):,}")
c2.metric("Realized over-2.5 rate", f"{covered['went_over'].mean():.1%}")
c3.metric("Market mean P(over)", f"{covered['p_over_Avg'].mean():.1%}")

st.subheader("Over rate by season: market vs reality")
per_season = covered.groupby("season").agg(
    realized=("went_over", "mean"), market=("p_over_Avg", "mean")
)
fig = go.Figure()
fig.add_trace(go.Scatter(x=per_season.index, y=per_season["realized"],
                         mode="lines+markers", name="realized over rate"))
fig.add_trace(go.Scatter(x=per_season.index, y=per_season["market"],
                         mode="lines+markers", name="market P(over)",
                         line=dict(dash="dot")))
fig.update_yaxes(tickformat=".0%")
fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Calibration")
book = st.selectbox("Book", books, format_func=TOTAL_BOOK_NAMES.get)
rel = reliability_binary(df, book)
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                         line=dict(dash="dash", color="gray")))
fig.add_trace(
    go.Scatter(
        x=rel["predicted"], y=rel["observed"], mode="markers+lines",
        name="observed",
        error_y=dict(type="data", symmetric=False,
                     array=rel["ci_hi"] - rel["observed"],
                     arrayminus=rel["observed"] - rel["ci_lo"]),
    )
)
fig.update_layout(
    xaxis=dict(title="implied P(over 2.5), de-margined", tickformat=".0%"),
    yaxis=dict(title="realized over rate", tickformat=".0%"),
    height=480,
)
st.plotly_chart(fig, use_container_width=True)
try:
    eff = efficiency_binary(df, book)
    e1, e2, e3 = st.columns(3)
    e1.metric("Slope", f"{eff['slope']:.3f} ± {eff['slope_se']:.3f}",
              help="1.0 = efficient")
    e2.metric("Intercept", f"{eff['intercept']:.3f} ± {eff['intercept_se']:.3f}")
    e3.metric("Matches", f"{eff['n']:,}")
except Exception:
    st.caption("Not enough data for the efficiency regression.")

st.subheader("Totals leaderboard")
common = df.dropna(subset=[f"p_over_{b}" for b in books])
st.markdown(f"Compared on **{len(common):,}** matches all books quote.")
rows = []
for b in books:
    ll = binary_log_loss(common[f"p_over_{b}"].to_numpy(), common["went_over"].to_numpy())
    mean, lo, hi = bootstrap_ci(ll)
    rows.append({"book": TOTAL_BOOK_NAMES[b], "log_loss": mean, "lo": lo, "hi": hi})
board = pd.DataFrame(rows).sort_values("log_loss")
fig = go.Figure(
    go.Scatter(
        x=board["log_loss"], y=board["book"], mode="markers", marker=dict(size=11),
        error_x=dict(type="data", symmetric=False,
                     array=board["hi"] - board["log_loss"],
                     arrayminus=board["log_loss"] - board["lo"]),
    )
)
fig.update_layout(
    xaxis_title="binary log loss (lower = sharper), 95% bootstrap CI",
    yaxis=dict(autorange="reversed"),
    height=90 + 55 * len(board),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Note the scale: binary log loss near ln(2) ≈ 0.693 means the market "
    "prices over/under close to a coin flip — which it is, by design of the "
    "2.5 line."
)
