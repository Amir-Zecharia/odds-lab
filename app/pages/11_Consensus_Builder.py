import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import BOOKS, load_matches, page_setup, season_filter
from src import config
from src.consensus import panel_universe, prob_cols, race
from src.evaluate import paired_bootstrap_diff

page_setup("Consensus Builder", icon="⚖️")
st.caption(
    "How should bookmakers be weighted in a prediction? Five schemes race "
    "on identical out-of-sample matches; ridge strength and softmax "
    "temperature are auto-tuned on held-out training seasons each refit."
)

df = season_filter(load_matches())

candidates = [
    b for b in BOOKS
    if b != "cons" and all(c in df.columns for c in prob_cols(b))
]
panel = st.multiselect(
    "Bookmaker panel",
    candidates,
    default=[b for b in ("PS", "B365", "WH", "Avg") if b in candidates],
    format_func=BOOKS.get,
    help="Schemes are fitted and compared only on matches where every "
    "panel book quotes the match, so all schemes see identical data.",
)
if len(panel) < 2:
    st.info("Pick at least two books.")
    st.stop()

universe = panel_universe(df, panel)
n_seasons = universe["season"].nunique()
st.markdown(
    f"**{len(universe):,} matches** where all {len(panel)} books quote "
    f"({universe['season'].min()} – {universe['season'].max()})."
)
if n_seasons < 5:
    st.warning("Fewer than 5 seasons of shared coverage — widen the seasons or drop a book.")
    st.stop()


@st.cache_data(show_spinner="Racing weighting schemes (expanding-window refits) ...")
def cached_race(seasons: tuple, panel_key: tuple):
    sub = df[df["season"].isin(seasons)]
    return race(sub, list(panel_key))


summary, losses, weights_df, tuning = cached_race(
    tuple(sorted(universe["season"].unique())), tuple(panel)
)

# ---- results -------------------------------------------------------------
st.subheader("The race")
baseline = "margin weights"
rows = []
for name in summary.index:
    if name == baseline:
        rows.append({"scheme": name, "diff": 0.0, "lo": 0.0, "hi": 0.0})
    else:
        boot = paired_bootstrap_diff(losses[name], losses[baseline])
        rows.append({"scheme": name, "diff": boot["diff"], "lo": boot["lo"], "hi": boot["hi"]})
diffs = pd.DataFrame(rows).set_index("scheme").loc[summary.index]

fig = go.Figure(
    go.Scatter(
        x=diffs["diff"], y=diffs.index, mode="markers", marker=dict(size=11),
        error_x=dict(type="data", symmetric=False,
                     array=diffs["hi"] - diffs["diff"],
                     arrayminus=diffs["diff"] - diffs["lo"]),
    )
)
fig.add_vline(x=0, line_dash="dash", line_color="gray")
fig.update_layout(
    xaxis_title=f"log-loss difference vs '{baseline}' "
    "(negative = better), 95% paired-bootstrap CI",
    yaxis=dict(autorange="reversed"),
    height=90 + 55 * len(diffs),
)
st.plotly_chart(fig, use_container_width=True)

table = summary.copy()
table["Δ vs margin"] = diffs["diff"]
st.dataframe(
    table.style.format({"log_loss": "{:.4f}", "Δ vs margin": "{:+.4f}"}),
    use_container_width=True,
)
st.caption(
    f"{summary['n_matches'].iloc[0]:,} out-of-sample matches. Auto-tuned on "
    f"the final refit: ridge C = {tuning['ridge_C']}, softmax temperature "
    f"= {tuning['temperature']}. A scheme genuinely improves on margin "
    "weighting only if its CI sits entirely left of zero."
)

# ---- the weights themselves ----------------------------------------------
st.subheader("The weights each scheme chose")
st.markdown(
    "From the final refit (training on all but the last test season). "
    "For ridge stacking the bars show *influence* — how much the fit leans "
    "on each book's price — rather than a mixing proportion."
)
melted = (
    weights_df.reset_index(names="scheme")
    .melt(id_vars="scheme", var_name="book", value_name="weight")
)
melted["book"] = melted["book"].map(BOOKS)
fig = px.bar(melted, x="book", y="weight", color="scheme", barmode="group")
fig.update_yaxes(tickformat=".0%")
fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True)

st.subheader("What the auto-search found, refit by refit")
st.markdown(
    "The optimizer searches the full weight simplex before every test "
    "season (the objective is convex, so each point is the exact global "
    "optimum given the data so far). If these lines jump around, the "
    "\"best\" weights are noise being chased; if they settle, the market "
    "has a stable hierarchy."
)
history = pd.DataFrame(tuning["weight_history"])
melted_h = history.melt(id_vars="test_season", var_name="book", value_name="weight")
melted_h["book"] = melted_h["book"].map(BOOKS)
fig = px.line(
    melted_h, x="test_season", y="weight", color="book", markers=True,
    labels={"test_season": "season being predicted"},
)
fig.update_yaxes(tickformat=".0%", range=[0, 1])
fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True)

st.subheader("How to read this")
st.markdown(
    """
- **If all CIs hug zero** (the expected outcome): books are so correlated
  that *any* sensible weighting lands within noise of any other. Use the
  simplest scheme — margin weights or even a plain average — and spend your
  effort elsewhere.
- **If “sharpest single quote” keeps up with blends**: one good book
  (Pinnacle closing where available) already contains the panel's
  information; blending adds robustness against a missing quote, not accuracy.
- **The auto-tuned ridge C is informative in itself**: a small C means the
  optimizer chose heavy shrinkage — it *couldn't* find stable weights better
  than near-equal ones, which is the collinearity story in one number.
"""
)
