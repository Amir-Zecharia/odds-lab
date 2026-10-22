import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import BOOKS, load_matches, page_setup, season_filter
from src import config
from src.evaluate import log_loss_per_match, unpaired_bootstrap_diff

page_setup("Round by Round", icon="📅")
st.caption(
    "Does bookmaker accuracy change across the season — sharper as team "
    "information accumulates — and has that pattern shifted over the years? "
    "Rounds are reconstructed from each team's match number in the season."
)

df = season_filter(load_matches())

EARLY, LATE = (1, 10), (29, 38)


def losses_for(frame: pd.DataFrame, book: str) -> pd.Series:
    cols = [f"p_{book}_{o}" for o in config.OUTCOMES]
    sub = frame.dropna(subset=cols)
    return pd.Series(
        log_loss_per_match(sub[cols].to_numpy(), sub["result"]), index=sub.index
    )


candidates = [
    b for b in BOOKS
    if all(f"p_{b}_{o}" in df.columns for o in config.OUTCOMES)
]

# ---- 1. Log loss by round, per book -----------------------------------------
st.subheader("Accuracy across the season")
chosen = st.multiselect(
    "Bookmakers",
    candidates,
    default=[b for b in ("cons", "PS", "B365", "Avg") if b in candidates],
    format_func=BOOKS.get,
)
if not chosen:
    st.info("Pick at least one bookmaker.")
    st.stop()

fig = go.Figure()
for book in chosen:
    ll = losses_for(df, book)
    per_round = (
        pd.DataFrame({"round": df.loc[ll.index, "round"], "ll": ll})
        .groupby("round")["ll"]
        .agg(["mean", "sem", "size"])
    )
    fig.add_trace(
        go.Scatter(
            x=per_round.index,
            y=per_round["mean"],
            error_y=dict(type="data", array=1.96 * per_round["sem"], thickness=0.8),
            mode="lines+markers",
            name=BOOKS[book],
            customdata=per_round["size"],
            hovertemplate="round %{x}: %{y:.4f} (n=%{customdata})",
        )
    )
fig.update_layout(
    xaxis_title="round (matchweek)",
    yaxis_title="mean log loss (lower = sharper)",
    height=460,
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Error bars are 95% intervals on the round mean. Expect the first rounds "
    "to sit higher: promoted teams, transfers and tactical changes make "
    "August genuinely harder to price than April."
)

# ---- 2. Early vs late verdict ------------------------------------------------
st.subheader("Early-season penalty")
book_one = st.selectbox("Bookmaker for the detail views", chosen, format_func=BOOKS.get)
ll_one = losses_for(df, book_one)
rounds_one = df.loc[ll_one.index, "round"]
early_ll = ll_one[rounds_one.between(*EARLY)]
late_ll = ll_one[rounds_one.between(*LATE)]
if len(early_ll) > 50 and len(late_ll) > 50:
    boot = unpaired_bootstrap_diff(early_ll.to_numpy(), late_ll.to_numpy())
    st.markdown(
        f"**{BOOKS[book_one]}**, rounds {EARLY[0]}–{EARLY[1]} vs "
        f"{LATE[0]}–{LATE[1]}: early-season log loss is "
        f"**{boot['diff']:+.4f}** vs late season "
        f"(95% CI [{boot['lo']:+.4f}, {boot['hi']:+.4f}]) — "
        + (
            "a real early-season penalty."
            if boot["lo"] > 0
            else "a real late-season penalty."
            if boot["hi"] < 0
            else "no significant difference."
        )
    )

# ---- 3. Stability across the years -------------------------------------------
st.subheader("Is the round pattern stable over the years?")
st.markdown(
    "Each cell: mean log loss of the chosen book in that season and phase "
    "of the campaign. A stable pattern shows as vertical bands; a changing "
    "one as horizontal drift."
)
bucket_edges = [0, 5, 10, 15, 20, 25, 30, 38]
bucket_labels = ["1–5", "6–10", "11–15", "16–20", "21–25", "26–30", "31–38"]
frame = pd.DataFrame(
    {
        "season": df.loc[ll_one.index, "season"],
        "bucket": pd.cut(rounds_one, bins=bucket_edges, labels=bucket_labels),
        "ll": ll_one,
    }
)
heat = frame.pivot_table(index="season", columns="bucket", values="ll", observed=True)
fig = px.imshow(
    heat,
    labels=dict(x="rounds", y="season", color="log loss"),
    color_continuous_scale="RdYlGn_r",
    aspect="auto",
)
fig.update_layout(height=620)
st.plotly_chart(fig, use_container_width=True)

per_season = frame.assign(
    phase=np.select(
        [frame["ll"].index.isin(early_ll.index), frame["ll"].index.isin(late_ll.index)],
        ["early (1–10)", "late (29–38)"],
        default="mid",
    )
)
per_season = (
    per_season[per_season["phase"] != "mid"]
    .groupby(["season", "phase"], observed=True)["ll"]
    .mean()
    .reset_index()
)
fig = px.line(
    per_season, x="season", y="ll", color="phase", markers=True,
    labels={"ll": "mean log loss"},
)
fig.update_layout(height=420, title=f"{BOOKS[book_one]}: early vs late season, year by year")
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "If the two lines converge over the years, the market has gotten better "
    "at pricing early-season uncertainty; parallel lines mean the penalty "
    "is structural. Year-to-year wiggle is mostly luck — 100 matches per "
    "point — so read the trend, not single seasons."
)
