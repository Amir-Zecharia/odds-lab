import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

from common import BOOKS, load_matches, page_setup, season_filter
from src.drift_analysis import (
    BOOK_PAIRS,
    SIGNIFICANT_QUANTILE,
    build_drift_frame,
    consensus_vs_closing,
    drift_surprise_regression,
    miss_rate_by_bucket,
    stability_over_time,
)

page_setup("Drift vs Surprises", icon="🚨")
st.caption(
    "When the odds move sharply before kickoff, does the result disagree "
    "with the bookmaker consensus more often? Miss = the consensus favorite "
    "does not win."
)
st.info(
    "**Coverage:** the source stores one price per match before 2012-13, so "
    "no early-vs-closing pair exists there. Pinnacle is the longest series "
    "(2012-13 onward, ~5,150 matches); other books pair up from 2019-20."
)

df = season_filter(load_matches())
book = st.selectbox(
    "Book pair (early → closing)",
    list(BOOK_PAIRS),
    format_func=lambda b: f"{BOOKS[b]} → {BOOKS[BOOK_PAIRS[b]]}",
)
frame = build_drift_frame(df, book, BOOK_PAIRS[book])
if len(frame) < 300:
    st.warning("Too few paired matches in this selection.")
    st.stop()

seasons = sorted(frame["season"].unique())
threshold = frame["drift"].quantile(SIGNIFICANT_QUANTILE)
c1, c2, c3 = st.columns(3)
c1.metric("Paired matches", f"{len(frame):,}")
c2.metric("Series", f"{seasons[0]} – {seasons[-1]}")
c3.metric(
    "Significant move ≥",
    f"{threshold:.1%}",
    help="Top decile of the max probability change across H/D/A.",
)

# ---- headline: miss rate by drift bucket -----------------------------------
st.subheader("Does the consensus favorite miss more when the line moves?")
table = miss_rate_by_bucket(frame)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=table["drift_mid"],
        y=table["miss_rate"],
        mode="markers+lines",
        name="favorite-miss rate",
        error_y=dict(
            type="data",
            symmetric=False,
            array=table["miss_hi"] - table["miss_rate"],
            arrayminus=table["miss_rate"] - table["miss_lo"],
        ),
        customdata=table["n"],
        hovertemplate="drift %{x:.1%}: miss %{y:.1%} (n=%{customdata})",
    )
)
fig.add_trace(
    go.Scatter(
        x=table["drift_mid"],
        y=1 - table["mean_fav_prob"],
        mode="lines",
        name="expected miss rate (1 − favorite prob)",
        line=dict(dash="dot", color="gray"),
    )
)
fig.update_layout(
    xaxis=dict(title="pre-close → close move size (max prob change)", tickformat=".1%"),
    yaxis=dict(title="consensus favorite-miss rate", tickformat=".0%"),
    height=460,
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "The gray dotted line is what the miss rate *should* be given how strong "
    "the favorites in each bucket are. Surprise-prediction means the markers "
    "sit systematically above that line as drift grows — not just both "
    "rising together."
)

reg = drift_surprise_regression(frame)
verdict = (
    "**Yes** — drift predicts consensus misses beyond favorite strength "
    f"(logit z = {reg['logit_drift_z']:+.2f}, p = {reg['logit_drift_p']:.3f})."
    if reg["logit_drift_p"] < 0.05
    else "**No** — controlling for favorite strength, drift does not "
    f"significantly predict consensus misses (z = {reg['logit_drift_z']:+.2f}, "
    f"p = {reg['logit_drift_p']:.3f}); the continuous error version agrees "
    f"(p = {reg['ols_drift_p']:.3f})."
)
st.markdown(f"Regression verdict on {reg['n']:,} matches: {verdict}")

# ---- who gets surprised ------------------------------------------------------
st.subheader("In big-move matches, who gets surprised?")
cvc = consensus_vs_closing(frame)
m1, m2, m3 = st.columns(3)
m1.metric("Big-move matches", f"{cvc['n_big_moves']:,}")
m2.metric("Consensus favorite misses", f"{cvc['consensus_fav_miss_rate']:.1%}")
m3.metric("Closing favorite misses", f"{cvc['closing_fav_miss_rate']:.1%}")
if cvc["n_favorites_disagree"] > 0:
    st.markdown(
        f"The move flipped the favorite in **{cvc['n_favorites_disagree']}** "
        "of those matches; where the two favorites disagreed, the closing "
        f"line's pick won **{cvc['disagreements_won_by_closing']:.0%}** of "
        "the time. The line moves toward truth — it just rarely moves far "
        "enough to change who is favored."
    )

# ---- stability over time ------------------------------------------------------
st.subheader("Is this stable over time?")
blocks = stability_over_time(frame)
if blocks.empty:
    st.caption("Not enough matches per period in this selection.")
else:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=blocks["block"], y=blocks["gap"],
                         customdata=blocks["n_big"],
                         hovertemplate="%{x}: gap %{y:+.1%} (n big=%{customdata})"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_yaxes(title="big-move miss rate − rest miss rate", tickformat="+.0%")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each bar: the extra miss rate of big-move matches vs everything "
        "else in that 3-season block. Persistent positive bars = a durable "
        "effect; sign-flipping bars = noise."
    )
