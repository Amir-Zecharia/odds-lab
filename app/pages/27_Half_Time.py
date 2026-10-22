import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import explain, load_matches, page_setup, season_filter
from src.halftime import (
    add_halftime_targets,
    comeback_rates,
    half_comparison,
    halftime_transition_matrix,
)

page_setup("Half Time", icon="⏱️")
st.caption(
    "Every match contains a second outcome the platform never looked at. "
    "Half-time scores have 100% coverage since 2000 — here is what they "
    "say about leads, comebacks, and whether the two halves are the "
    "same game."
)

df = add_halftime_targets(season_filter(load_matches()))
comparison = half_comparison(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matches", f"{comparison['n']:,}")
c2.metric(
    "Goals: 1st vs 2nd half",
    f"{comparison['first_half_goals']:.2f} → {comparison['second_half_goals']:.2f}",
    help="Second halves are consistently more productive.",
)
c3.metric(
    "Draw rate: HT → FT",
    f"{comparison['ht_draw_rate']:.0%} → {comparison['ft_draw_rate']:.0%}",
)
c4.metric("Both teams scored", f"{comparison['btts_rate']:.1%}")

# ---- transitions ---------------------------------------------------------------
st.subheader("Does a half-time lead hold?")
matrix = halftime_transition_matrix(df)
labels = {"H": "Home ahead", "D": "Level", "A": "Away ahead"}
display = matrix.rename(index=labels, columns={"H": "Home win", "D": "Draw", "A": "Away win"})
fig = px.imshow(
    display,
    labels=dict(x="full time", y="at half time", color="probability"),
    color_continuous_scale="Blues",
    text_auto=".1%",
    aspect="auto",
)
fig.update_layout(height=380)
st.plotly_chart(fig, use_container_width=True)

rates = comeback_rates(df)
home_flip = (
    rates["home_led_and_lost"] / rates["home_led_at_ht"]
    if rates["home_led_at_ht"] else float("nan")
)
away_flip = (
    rates["away_led_and_lost"] / rates["away_led_at_ht"]
    if rates["away_led_at_ht"] else float("nan")
)
m1, m2, m3 = st.columns(3)
m1.metric("Leads overturned", f"{rates['comeback_rate']:.1%}",
          help=f"{rates['leads_overturned']:,} of {rates['matches_with_a_lead']:,} "
          "half-time leads ended in defeat.")
m2.metric("Home lead lost", f"{home_flip:.1%}")
m3.metric("Away lead lost", f"{away_flip:.1%}",
          help="Away leads are markedly more fragile — home advantage shows "
          "up as the ability to come back.")
if home_flip == home_flip and away_flip == away_flip and home_flip > 0:
    st.markdown(
        f"An away side leading at the break loses **{away_flip / home_flip:.1f}×** "
        "as often as a home side in the same position. Home advantage is not "
        "only about scoring first — it is about recovering."
    )

# ---- HT/FT states ----------------------------------------------------------------
st.subheader("Half-time / full-time combinations")
states = df.dropna(subset=["ht_ft"])["ht_ft"].value_counts(normalize=True).sort_index()
fig = px.bar(
    states.reset_index(), x="ht_ft", y="proportion" if "proportion" in states.reset_index() else states.name or 0,
    labels={"ht_ft": "half time / full time", "y": "share of matches"},
)
fig.update_yaxes(tickformat=".1%")
fig.update_layout(height=380)
st.plotly_chart(fig, use_container_width=True)

# ---- over time ---------------------------------------------------------------------
st.subheader("Are the halves changing?")
by_season = df.dropna(subset=["ht_result"]).groupby("season").agg(
    first_half=("ht_goals", "mean"),
    second_half=("second_half_goals", "mean"),
    btts=("btts", "mean"),
)
fig = go.Figure()
fig.add_trace(go.Scatter(x=by_season.index, y=by_season["first_half"],
                         mode="lines+markers", name="1st-half goals"))
fig.add_trace(go.Scatter(x=by_season.index, y=by_season["second_half"],
                         mode="lines+markers", name="2nd-half goals"))
fig.update_layout(height=400, yaxis_title="goals per match")
st.plotly_chart(fig, use_container_width=True)

explain(
    "How to read this page",
    """
- **Two outcomes per match.** Half-time turns one 1X2 result into a
  second, differently-shaped one: level scores are far more common after
  45 minutes (~42%) than after 90 (~26%), because there has been less
  time for the favourite to convert its edge.
- **Second halves are livelier.** Roughly a quarter more goals arrive
  after the break — tiring legs, chasing teams, and substitutions.
- **Leads are strong but not safe.** Around 8% of half-time leads end in
  defeat, and the split by side is the interesting part: away leads are
  overturned about twice as often as home ones. That asymmetry *is* home
  advantage, expressed as recovery rather than dominance.
- **No market is involved here.** These are structural facts about
  football, measured over the full 26-season dataset, and they set a
  baseline any half-time forecast has to beat.
""",
)
