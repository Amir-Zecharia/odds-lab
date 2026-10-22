import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import BOOKS, explain, load_matches, page_setup, season_filter
from src.influence import (
    PAIRED_BOOKS,
    convergence_regression,
    follow_the_sharp,
    leadership_matrix,
)

page_setup("Bookmaker Influence", icon="🐑")
st.caption(
    "Do bookmakers change their odds because of other bookmakers? "
    "Who leads, who follows, and how strongly the pack pulls."
)
st.info(
    "**Resolution caveat:** the data holds two snapshots per book (early "
    "week + closing), not tick-by-tick series. \"A leads B\" here means "
    "A's earlier prices contain information B's closing prices later "
    "incorporate — price discovery, not literal minute-level reaction."
)

df = season_filter(load_matches())
books = [b for b in PAIRED_BOOKS if f"p_{PAIRED_BOOKS[b]}_H" in df.columns]

# ---- herding ----------------------------------------------------------------
st.subheader("Herding: who corrects toward the pack?")
st.markdown(
    "Each book's early→close move, regressed on how far it stood from the "
    "cross-book consensus at the early snapshot. **1.0 = fully erases its "
    "disagreement with the pack by kickoff; 0 = prices independently.**"
)
conv = pd.DataFrame([convergence_regression(df, b) for b in books])
conv["book"] = conv["book"].map(BOOKS)
fig = go.Figure(
    go.Bar(
        x=conv["book"], y=conv["herding_coef"],
        error_y=dict(type="data", array=1.96 * conv["se"]),
        marker_color=["#4c78a8" if c < 0.4 else "#e45756" for c in conv["herding_coef"]],
        customdata=conv["n"],
        hovertemplate="%{x}: %{y:.2f} (n=%{customdata:,})",
    )
)
fig.update_yaxes(title="herding coefficient", range=[0, 1])
fig.update_layout(height=380)
st.plotly_chart(fig, use_container_width=True)

# ---- leadership matrix --------------------------------------------------------
st.subheader("Who leads whom?")
st.markdown(
    "For every ordered pair: does the **leader's early** price explain the "
    "**follower's closing** price beyond the follower's own early price? "
    "Cell = that cross-coefficient (deeper red = stronger leadership)."
)
matrix = leadership_matrix(df, books)
pivot = matrix.pivot(index="leader", columns="follower", values="coef")
pivot.index = [BOOKS[b] for b in pivot.index]
pivot.columns = [BOOKS[b] for b in pivot.columns]
fig = px.imshow(
    pivot,
    color_continuous_scale="Reds",
    labels=dict(x="follower (close)", y="leader (early)", color="influence"),
    text_auto=".2f",
    aspect="auto",
)
fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True)

z_table = matrix.assign(
    leader=matrix["leader"].map(BOOKS), follower=matrix["follower"].map(BOOKS)
)
st.dataframe(
    z_table.style.format({"coef": "{:+.2f}", "z": "{:+.1f}"}),
    use_container_width=True,
)

# ---- follow the sharp ----------------------------------------------------------
st.subheader("Following the sharp")
rows = [follow_the_sharp(df, b) for b in books if b != "PS"]
if rows:
    fts = pd.DataFrame(rows)
    for _, r in fts.iterrows():
        st.markdown(
            f"- **{BOOKS[r['soft_book']]}**: a 1-unit gap to Pinnacle's early "
            f"position pulls its own move by **{r['gap_coef']:+.2f}** "
            f"(t = {r['gap_t']:+.1f}); the gap alone explains "
            f"**{r['move_r_squared']:.0%}** of its price changes "
            f"(n = {r['n']:,})."
        )

explain(
    "How to read this page",
    """
- **Herding chart:** Pinnacle's ~0.1 means it mostly ignores where other
  books stand — it prices from its own model and its bettors. Bet365 and
  William Hill at ~0.7 erase most of their disagreement with the pack by
  kickoff: strong evidence they *do* change odds because of other books.
- **Leadership matrix:** rows are the book whose early price does the
  explaining. The Pinnacle row glows and the Pinnacle *column* is dark —
  information flows one way. Within the followers, Bet365 leads William
  Hill more than the reverse: a full pecking order.
- **Why it works this way:** Pinnacle accepts huge stakes from winning
  players, so its price *is* aggregated information; recreational books
  manage risk by anchoring to it rather than out-forecasting it.
- Everything is computed on your current league/season selection — try
  a single league or era to see how stable the hierarchy is.
""",
)
