import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import explain, league_selector, load_matches, page_setup
from src import config
from src.models import DixonColes
from src.season_sim import (
    league_table,
    position_distribution,
    simulate_season,
    summarize_simulations,
)

page_setup("Season Simulator", icon="🔮")
st.info("Retrospective historical replay using the stored snapshot, not live fixtures or a current-season forecast.")
st.caption(
    "Who wins the league? Who goes down? Play the rest of a season "
    "thousands of times from per-match probabilities and count how often "
    "each story happens. No market involved — pure forecasting."
)

df = load_matches()
league = league_selector()
if league == config.ALL_LEAGUES:
    st.info("Pick a single league in the sidebar — a table needs one competition.")
    st.stop()
league_df = df[df["league"] == league].sort_values("date")

seasons = sorted(league_df["season"].unique())
season = st.selectbox("Season", list(reversed(seasons)))
season_df = league_df[league_df["season"] == season]

c1, c2 = st.columns(2)
matchweek = c1.slider(
    "Simulate from after this fraction of the season",
    0.0, 1.0, 0.6, 0.05,
    help="0.0 = predict the whole season before a ball is kicked; "
    "0.9 = only the run-in remains.",
)
n_sims = c2.select_slider("Simulations", [200, 500, 1000, 2000, 5000], value=1000)

cutoff = int(len(season_df) * matchweek)
so_far, rest = season_df.iloc[:cutoff], season_df.iloc[cutoff:]
if rest.empty:
    st.warning("Nothing left to simulate — lower the slider.")
    st.stop()


@st.cache_data(show_spinner="Fitting the goal model and simulating ...")
def _simulate(league: str, season: str, cutoff: int, n_sims: int, dataset_mtime_ns: int):
    data = load_matches()
    data = data[data["league"] == league].sort_values("date")
    current = data[data["season"] == season]
    train = data[data["season"] < season].tail(2500)
    if len(train) < 500:
        return None
    model = DixonColes().fit(train)
    played, remaining = current.iloc[:cutoff], current.iloc[cutoff:]
    probs = np.vstack(
        [model.outcome_probs(r.home_team, r.away_team) for r in remaining.itertuples()]
    )
    tables = simulate_season(
        played, remaining[["home_team", "away_team"]], probs,
        n_sims=n_sims, seed=7,
    )
    return tables, played, remaining


result = _simulate(league, season, cutoff, n_sims, (config.PROCESSED_DIR / "enriched.parquet").stat().st_mtime_ns)
if result is None:
    st.warning("Not enough prior seasons in this league to train the model.")
    st.stop()
tables, played, remaining = result

st.markdown(
    f"**{len(played)} matches played, {len(remaining)} simulated** "
    f"{n_sims:,} times each."
)

n_teams = len(tables[0])
relegation = 3 if n_teams >= 18 else 2
summary = summarize_simulations(tables, european_places=4, relegation_places=relegation)

st.subheader("Where the season ends")
show = summary.copy()
st.dataframe(
    show.style.format(
        {"expected_points": "{:.1f}", "points_lo": "{:.0f}", "points_hi": "{:.0f}",
         "expected_rank": "{:.1f}", "title": "{:.1%}", "european": "{:.1%}",
         "relegation": "{:.1%}"}
    ),
    use_container_width=True,
    height=min(80 + 35 * len(show), 700),
)

st.subheader("Full finishing-position distribution")
dist = position_distribution(tables)
dist = dist.loc[summary.index]
fig = px.imshow(
    dist,
    labels=dict(x="finishing position", y="", color="probability"),
    color_continuous_scale="Blues",
    aspect="auto",
)
fig.update_layout(height=60 + 28 * len(dist))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Points: expected, with 90% range")
fig = go.Figure(
    go.Scatter(
        x=summary["expected_points"], y=summary.index, mode="markers",
        marker=dict(size=10),
        error_x=dict(type="data", symmetric=False,
                     array=summary["points_hi"] - summary["expected_points"],
                     arrayminus=summary["expected_points"] - summary["points_lo"]),
    )
)
fig.update_layout(height=60 + 26 * len(summary), yaxis=dict(autorange="reversed"),
                  xaxis_title="final points")
st.plotly_chart(fig, use_container_width=True)

# Hindsight check — only meaningful for completed seasons.
if len(remaining) and season_df["home_goals"].notna().all():
    actual = league_table(season_df)
    st.subheader("How it actually finished")
    champion = actual.index[0]
    called = summary["title"].idxmax()
    st.markdown(
        f"Champion: **{champion}** — the simulation gave them "
        f"**{summary.loc[champion, 'title']:.1%}** "
        f"(its favourite was {called} at {summary.loc[called, 'title']:.1%})."
    )
    comparison = pd.DataFrame(
        {
            "simulated_points": summary["expected_points"],
            "actual_points": actual["points"],
            "actual_rank": actual["rank"],
        }
    ).dropna()
    comparison["error"] = comparison["simulated_points"] - comparison["actual_points"]
    st.dataframe(
        comparison.sort_values("actual_rank").style.format(
            {"simulated_points": "{:.1f}", "actual_points": "{:.0f}",
             "actual_rank": "{:.0f}", "error": "{:+.1f}"}
        ),
        use_container_width=True,
    )
    st.caption(
        f"Mean absolute points error: "
        f"**{comparison['error'].abs().mean():.1f}** across {len(comparison)} teams."
    )

explain(
    "How to read this page",
    """
- **The method.** A Dixon-Coles goal model is fitted on the seasons
  *before* the one you're viewing, then every remaining fixture is played
  out thousands of times. Counting the outcomes turns match probabilities
  into league probabilities.
- **Why the slider matters.** Drag it left and you're forecasting from
  near-zero information — title races look wide open. Drag it right and
  the table has mostly decided itself. Watching a team's title
  probability move as the season progresses is the point.
- **Nominal scorelines.** Simulated matches use a 2-0 / 1-1 convention:
  points are exactly right, goal difference is only a sensible
  tiebreaker. Positions decided by a single goal are approximate.
- **This is the honest kind of prediction.** No market prices league
  outcomes here, so nothing is being "beaten" — the forecast stands on
  its own. For completed seasons the page shows you exactly how wrong it
  was.
""",
)
