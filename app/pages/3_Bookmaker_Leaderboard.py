import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import BOOKS, load_matches, page_setup, season_filter
from src import config
from src.evaluate import bootstrap_ci, brier_per_match, log_loss_per_match

page_setup("Bookmaker Leaderboard", icon="🏆")
st.caption(
    "Which book's (Shin de-margined) probabilities score best? Books are "
    "compared only on matches they all cover. Expected winner: Pinnacle."
)

df = season_filter(load_matches())

available = [
    b for b in BOOKS
    if all(f"p_{b}_{o}" in df.columns for o in config.OUTCOMES)
]
chosen = st.multiselect(
    "Books to compare",
    available,
    default=[b for b in ("PS", "PSC", "B365", "Avg", "Max", "cons") if b in available],
    format_func=BOOKS.get,
)
if len(chosen) < 2:
    st.info("Pick at least two books.")
    st.stop()

prob_cols = {b: [f"p_{b}_{o}" for o in config.OUTCOMES] for b in chosen}
common = df.dropna(subset=[c for cols in prob_cols.values() for c in cols])
st.markdown(
    f"**{len(common):,} matches** covered by every selected book "
    f"({common['season'].min()} – {common['season'].max()})."
)
if common.empty:
    st.warning("No overlapping coverage — deselect a book (closing lines start 2019-20).")
    st.stop()

rows = []
for book in chosen:
    probs = common[prob_cols[book]].to_numpy()
    ll = log_loss_per_match(probs, common["result"])
    bs = brier_per_match(probs, common["result"])
    ll_mean, ll_lo, ll_hi = bootstrap_ci(ll)
    rows.append(
        {
            "book": BOOKS[book],
            "log_loss": ll_mean,
            "ll_lo": ll_lo,
            "ll_hi": ll_hi,
            "brier": bs.mean(),
        }
    )
board = pd.DataFrame(rows).sort_values("log_loss")

fig = go.Figure(
    go.Scatter(
        x=board["log_loss"],
        y=board["book"],
        error_x=dict(
            type="data",
            symmetric=False,
            array=board["ll_hi"] - board["log_loss"],
            arrayminus=board["log_loss"] - board["ll_lo"],
        ),
        mode="markers",
        marker=dict(size=11),
    )
)
fig.update_layout(
    xaxis_title="log loss (lower is sharper) with 95% bootstrap CI",
    yaxis=dict(autorange="reversed"),
    height=90 + 60 * len(board),
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    board.set_index("book").style.format("{:.4f}"),
    use_container_width=True,
)
st.caption(
    "The differences are small in absolute terms — every serious book is "
    "within a few thousandths of a nat. That closeness is the finding."
)
