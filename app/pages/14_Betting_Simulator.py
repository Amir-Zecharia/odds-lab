import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import streamlit as st

from common import BOOKS, load_matches, page_setup, season_filter
from src.simulator import (
    bet_favorite,
    bet_home,
    bet_longshot,
    bet_value,
    summarize,
)

page_setup("Betting Simulator", icon="🎰")
st.caption(
    "Watch the margin grind. Backtest a rule against two decades of real "
    "prices — the point of this page is understanding *why* near-perfect "
    "probabilities still lose money. Educational only."
)

df = season_filter(load_matches())

col1, col2 = st.columns(2)
strategy = col1.selectbox(
    "Strategy",
    ["always the favorite", "always the home team", "always the longshot",
     "value vs model"],
)
book = col2.selectbox(
    "Execute at",
    [b for b in ("B365", "Avg", "Max", "PS") if f"{b}H" in df.columns],
    format_func=lambda b: f"{BOOKS[b]} prices",
)

if strategy == "value vs model":
    col3, col4, col5 = st.columns(3)
    prob_source = col3.selectbox(
        "Model probabilities",
        [p for p in ("cons", "PSC", "PS", "Avg") if f"p_{p}_H" in df.columns],
        format_func=BOOKS.get,
    )
    edge = col4.slider("Minimum edge", 0.0, 0.20, 0.02, 0.01, format="%.2f",
                       help="Bet only when model probability x odds > 1 + edge")
    kelly = col5.toggle("Kelly staking", value=False,
                        help="Stake a capped Kelly fraction of a 100-unit "
                        "bankroll instead of flat 1 unit")
    bets = bet_value(df, book, prob_source, edge, kelly)
else:
    fn = {"always the favorite": bet_favorite,
          "always the home team": bet_home,
          "always the longshot": bet_longshot}[strategy]
    bets = fn(df, book)

if bets.empty:
    st.warning("This rule places no bets on the current selection.")
    st.stop()

s = summarize(bets)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Bets placed", f"{s['bets']:,}")
m2.metric("Hit rate", f"{s['hit_rate']:.1%}")
m3.metric("Total P&L", f"{s['profit']:+,.1f} units")
m4.metric("ROI", f"{s['roi']:+.2%}")

bets = bets.sort_values("date")
bets["bankroll"] = bets["profit"].cumsum()
fig = px.line(bets, x="date", y="bankroll",
              labels={"bankroll": "cumulative profit (units)"})
fig.add_hline(y=0, line_dash="dash", line_color="gray")
fig.update_layout(height=440)
st.plotly_chart(fig, use_container_width=True)

per_season = bets.groupby("season").agg(
    bets=("profit", "size"), profit=("profit", "sum"),
    staked=("stake", "sum"),
)
per_season["roi"] = per_season["profit"] / per_season["staked"]
fig = px.bar(per_season.reset_index(), x="season", y="roi",
             labels={"roi": "return on stakes"})
fig.update_yaxes(tickformat="+.0%")
fig.update_layout(height=340)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"""
**Why it looks like this.** The market's probabilities are calibrated
(slope ≈ 1.04), so no mechanical rule finds mispriced outcomes — but every
bet pays the bookmaker margin. At {BOOKS[book]} prices the margin has
averaged a few percent per bet, so the expected ROI of *any* rule sits
near minus the margin. "Value vs model" using the market's own consensus
mostly picks up noise plus the best-price edge; watch how the Kelly toggle
amplifies variance without changing the sign of the drift.
"""
)
