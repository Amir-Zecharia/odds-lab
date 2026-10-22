import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import explain, league_selector, load_matches, page_setup
from src import config
from src.forecast import forecast_match, market_view

page_setup("Match Forecast", icon="🎱")
st.info("Retrospective historical replay using the stored snapshot, not live fixtures or a current-season forecast.")
st.caption(
    "One fixture, everything the platform knows: outcome, goals and "
    "scorelines, corners and cards adjusted for the referee, and how the "
    "first half is likely to unfold. Ten studies pointed at one match."
)

df = load_matches()
league = league_selector()
if league == config.ALL_LEAGUES:
    st.info("Pick a single league in the sidebar — a forecast needs one competition.")
    st.stop()
league_df = df[df["league"] == league].sort_values("date").reset_index(drop=True)

seasons = sorted(league_df["season"].unique())
season = st.selectbox("Season", list(reversed(seasons)))
season_df = league_df[league_df["season"] == season]

fixtures = season_df.sort_values("date", ascending=False)
labels = {
    int(r.match_id): f"{r.date.date()} · {r.home_team} vs {r.away_team}"
    for r in fixtures.itertuples()
}
match_id = st.selectbox("Fixture", list(labels), format_func=labels.get)
row = league_df[league_df["match_id"] == match_id].iloc[0]

# Strict discipline: the model may only see matches that finished earlier.
history = league_df[league_df["date"] < row["date"]]
if len(history) < 500:
    st.warning("Not enough earlier matches in this league to fit the models.")
    st.stop()
history = history.tail(3000)

referee = row.get("referee")
referee = referee if isinstance(referee, str) and referee.strip() else None


@st.cache_data(show_spinner="Fitting every model on matches before kickoff ...")
def _forecast(match_id: int, league: str, referee: str | None, dataset_mtime_ns: int):
    data = load_matches()
    data = data[data["league"] == league].sort_values("date").reset_index(drop=True)
    target = data[data["match_id"] == match_id].iloc[0]
    past = data[data["date"] < target["date"]].tail(3000)
    return forecast_match(
        past, target["home_team"], target["away_team"],
        referee=referee, market=market_view(target),
    )


f = _forecast(int(match_id), league, referee, (config.PROCESSED_DIR / "enriched.parquet").stat().st_mtime_ns)

st.subheader(f"{f.home_team} vs {f.away_team}")
if referee:
    st.caption(f"Referee: **{referee}** — card forecast adjusted for their tendency.")

# ---- outcome -------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric(f"{f.home_team} win", f"{f.p_home:.1%}")
c2.metric("Draw", f"{f.p_draw:.1%}")
c3.metric(f"{f.away_team} win", f"{f.p_away:.1%}")

if f.market:
    st.markdown("**Our model vs the market** (the market is the benchmark here).")
    compare = pd.DataFrame(
        {
            "model": [f.p_home, f.p_draw, f.p_away],
            f"market ({f.market['source']})": [
                f.market["p_home"], f.market["p_draw"], f.market["p_away"]
            ],
        },
        index=["Home", "Draw", "Away"],
    )
    fig = px.bar(compare, barmode="group", labels={"value": "probability", "index": ""})
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=330)
    st.plotly_chart(fig, use_container_width=True)
    biggest = (compare["model"] - compare.iloc[:, 1]).abs().idxmax()
    st.caption(
        f"Largest disagreement on **{biggest}**. Every study here says the "
        "market is the better forecast — where they differ, trust the market."
    )

# ---- goals ---------------------------------------------------------------------
st.subheader("Goals")
g1, g2, g3, g4 = st.columns(4)
g1.metric("Expected goals", f"{f.expected_home_goals + f.expected_away_goals:.2f}",
          help=f"{f.expected_home_goals:.2f} – {f.expected_away_goals:.2f}")
g2.metric("Over 2.5", f"{f.p_over_25_goals:.1%}")
g3.metric("Both teams score", f"{f.p_btts:.1%}")
g4.metric("Most likely score", f.top_scorelines[0]["score"],
          help=f"{f.top_scorelines[0]['prob']:.1%}")

scores = pd.DataFrame(f.top_scorelines)
fig = px.bar(scores, x="score", y="prob", labels={"prob": "probability", "score": ""})
fig.update_yaxes(tickformat=".1%")
fig.update_layout(height=320)
st.plotly_chart(fig, use_container_width=True)

# ---- counts ----------------------------------------------------------------------
def display_count(value, spec=".1f"):
    return format(value, spec) if pd.notna(value) else "Unavailable"


def actual_total(home, away):
    return str(int(home + away)) if pd.notna(home) and pd.notna(away) else "Unavailable"


st.subheader("Corners and cards")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Expected corners", display_count(f.expected_corners, ".1f"))
k2.metric(f"Over {f.corner_line}", display_count(f.p_over_corners_line, ".1%"))
k3.metric("Expected cards", display_count(f.expected_cards, ".1f"))
k4.metric(f"Over {f.card_line}", display_count(f.p_over_cards_line, ".1%"))
st.caption(
    "These carry the weakest signal on the platform — team tendencies in "
    "corners and cards barely beat a flat league average. Treat them as "
    "informed baselines, not edges."
)

# ---- half time ---------------------------------------------------------------------
st.subheader("Half time")
ht = f.half_time
h1, h2, h3 = st.columns(3)
h1.metric(f"{f.home_team} ahead", f"{ht['H']:.1%}")
h2.metric("Level", f"{ht['D']:.1%}")
h3.metric(f"{f.away_team} ahead", f"{ht['A']:.1%}")
if f.ht_to_ft:
    trans = pd.DataFrame(f.ht_to_ft).T.reindex(config.OUTCOMES)
    trans.index = ["Home ahead", "Level", "Away ahead"]
    trans.columns = ["Home win", "Draw", "Away win"]
    fig = px.imshow(trans, text_auto=".0%", color_continuous_scale="Blues",
                    labels=dict(x="full time", y="at half time", color="P"))
    fig.update_layout(height=330)
    st.plotly_chart(fig, use_container_width=True)

# ---- what actually happened ---------------------------------------------------------
if pd.notna(row.get("home_goals")):
    st.subheader("What actually happened")
    actual = f"{int(row['home_goals'])}-{int(row['away_goals'])}"
    hit = next((s["prob"] for s in f.top_scorelines if s["score"] == actual), None)
    a1, a2, a3 = st.columns(3)
    a1.metric("Final score", actual)
    a2.metric("Result", {"H": "Home win", "D": "Draw", "A": "Away win"}[row["result"]],
              help=f"Model gave it "
              f"{ {'H': f.p_home, 'D': f.p_draw, 'A': f.p_away}[row['result']]:.1%}")
    a3.metric("Actual corners / cards",
              f"{actual_total(row.get('home_corners'), row.get('away_corners'))} / "
              f"{actual_total(row.get('home_yellows'), row.get('away_yellows'))}")
    if hit:
        st.caption(f"That exact scoreline was in the model's top six, at {hit:.1%}.")

explain(
    "How to read this page",
    """
- **Everything here is fitted on matches that finished before kickoff.**
  The page re-fits from scratch for the fixture you pick, using at most
  the previous 3,000 matches in that league — so a historical fixture is
  forecast exactly as it would have been at the time.
- **Where model and market disagree, the market is right.** That is not
  modesty; it is the repeatedly-measured finding of this platform. The
  model's independent opinion is shown because the *difference* is
  interesting, not because it is better.
- **Confidence varies wildly by market.** The outcome and goals forecasts
  rest on Dixon-Coles, which is well validated. The corner and card
  numbers rest on tendencies that barely beat a league average — they are
  baselines, not predictions with an edge.
- **Half time is derived**, not separately modelled: roughly 42% of goals
  arrive before the break, so the half is the same goal model scaled.
  The transition grid beneath it is measured from all 61k matches.
""",
)
