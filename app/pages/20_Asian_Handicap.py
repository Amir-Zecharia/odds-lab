import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import explain, load_matches, page_setup, season_filter
from src.asian_handicap import (
    AH_BOOK_NAMES,
    _truth_col,
    available_ah_books,
    efficiency_ah,
    leaderboard_ah,
    reliability_ah,
)

page_setup("Asian Handicap", icon="⚖️")
st.caption(
    "The two-way handicap market: no draw, and a line chosen so the price "
    "sits near even money at any matchup. The cleanest test of whether the "
    "market's implied probabilities are truly calibrated."
)

df = season_filter(load_matches())
books = available_ah_books(df)
if not books:
    st.warning(
        "No Asian-handicap odds in the selected seasons. Re-run "
        "`python -m src.ingest` then `python -m src.run_analyses` to ingest them."
    )
    st.stop()

covered = df.dropna(subset=["p_ahcover_Avg"]) if "p_ahcover_Avg" in df.columns else df
decisive = covered.dropna(subset=["ah_covered"])
c1, c2, c3 = st.columns(3)
c1.metric("Matches with AH odds", f"{len(covered):,}")
c2.metric("Realized home-cover rate", f"{decisive['ah_covered'].mean():.1%}")
c3.metric("Market mean P(home covers)", f"{covered['p_ahcover_Avg'].mean():.1%}")

st.subheader("Home-cover rate by season: market vs reality")
per_season = decisive.groupby("season").agg(
    realized=("ah_covered", "mean"), market=("p_ahcover_Avg", "mean")
)
fig = go.Figure()
fig.add_trace(go.Scatter(x=per_season.index, y=per_season["realized"],
                         mode="lines+markers", name="realized cover rate"))
fig.add_trace(go.Scatter(x=per_season.index, y=per_season["market"],
                         mode="lines+markers", name="market P(cover)",
                         line=dict(dash="dot")))
fig.update_yaxes(tickformat=".0%")
fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Calibration")
book = st.selectbox("Book", books, format_func=AH_BOOK_NAMES.get)
rel = reliability_ah(df, book)
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
    xaxis=dict(title="implied P(home covers), de-margined", tickformat=".0%"),
    yaxis=dict(title="realized cover rate", tickformat=".0%"),
    height=480,
)
st.plotly_chart(fig, use_container_width=True)
try:
    eff = efficiency_ah(df, book)
    e1, e2, e3 = st.columns(3)
    e1.metric("Slope", f"{eff['slope']:.3f} ± {eff['slope_se']:.3f}",
              help="1.0 = efficient (implied probability is the true probability)")
    e2.metric("Intercept", f"{eff['intercept']:.3f} ± {eff['intercept_se']:.3f}")
    e3.metric("Decisive bets", f"{eff['n']:,}")
except Exception:
    st.caption("Not enough data for the efficiency regression.")

st.subheader("Asian-handicap leaderboard")
common = df.dropna(subset=[f"p_ahcover_{b}" for b in books])
st.markdown(f"Compared on **{len(common):,}** matches all books quote.")
board = leaderboard_ah(df, books)
fig = go.Figure(
    go.Scatter(
        x=board["log_loss"], y=board["name"], mode="markers", marker=dict(size=11),
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
    "Log loss near ln(2) ≈ 0.693 is expected: the handicap line is set so "
    "each side is close to a 50/50 shot. What matters is calibration across "
    "the range, not the average level."
)

explain(
    "How to read this",
    """
**The market.** An Asian handicap gives the home team a head start (or
deficit) — the *line* `AHh`, applied to the goal difference. A bet on the
home side at line `L` settles on `goal_diff + L`: positive is a win,
negative a loss, exactly zero a **push** (stake returned).

**Quarter lines.** A line like `-0.75` is settled as two equal half-stakes
at the neighbouring `-0.5` and `-1.0` lines. Winning by one goal then wins
the `-0.5` half and pushes the `-1.0` half → a **half-win** (+0.5). This
yields settlement fractions in {-1, -0.5, 0, +0.5, +1}. For calibration we
drop exact pushes and treat any positive settlement as "home covered".

**Why AH is the cleanest efficiency test.** The 1X2 market has three
outcomes and the over/under 2.5 line sits near a coin flip, leaving little
probability range to calibrate. Asian handicaps remove the draw and the
bookmaker picks the line so the two-way price hovers near even money at
*any* matchup — spreading the de-margined P(home covers) across the whole
0–1 range. A logistic fit of covered on `logit(P)` with **slope 1,
intercept 0** means the price is the true probability. Deviations are pure
miscalibration, with no draw to absorb the error.
""",
)
