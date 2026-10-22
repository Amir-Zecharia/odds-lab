import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common import BOOKS, load_matches, page_setup, season_filter
from src import config

page_setup("Descriptive Statistics", icon="🧮")
st.caption(
    "Slice the dataset any way you like — every number below recomputes "
    "from the matches passing your filters."
)

df = season_filter(load_matches())

# ---- Filters -----------------------------------------------------------------
st.sidebar.subheader("Filters")

teams = sorted(set(df["home_team"]) | set(df["away_team"]))
picked_teams = st.sidebar.multiselect("Teams", teams)
venue = st.sidebar.radio("Venue for picked teams", ["Either", "Home only", "Away only"])
if picked_teams:
    if venue == "Home only":
        df = df[df["home_team"].isin(picked_teams)]
    elif venue == "Away only":
        df = df[df["away_team"].isin(picked_teams)]
    else:
        df = df[df["home_team"].isin(picked_teams) | df["away_team"].isin(picked_teams)]

outcomes = st.sidebar.multiselect(
    "Result", ["H", "D", "A"],
    format_func={"H": "Home win", "D": "Draw", "A": "Away win"}.get,
)
if outcomes:
    df = df[df["result"].isin(outcomes)]

months = st.sidebar.multiselect(
    "Months", list(range(1, 13)),
    format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"),
)
if months:
    df = df[df["date"].dt.month.isin(months)]

source = st.sidebar.selectbox(
    "Probability source",
    [b for b in ("cons", "PS", "PSC", "B365", "Avg", "Max")
     if f"p_{b}_H" in df.columns],
    format_func=BOOKS.get,
)
p_cols = [f"p_{source}_{o}" for o in config.OUTCOMES]
fav_prob_all = df[p_cols].max(axis=1)
lo, hi = st.sidebar.slider(
    "Favorite's implied probability", 0.30, 1.0, (0.30, 1.0), step=0.05,
    help="Keep only matches whose market favorite falls in this range "
    "(e.g. 0.30–0.45 = tight matches, 0.70+ = heavy favorites).",
)
if (lo, hi) != (0.30, 1.0):
    df = df[fav_prob_all.between(lo, hi)]

if df.empty:
    st.warning("No matches pass these filters.")
    st.stop()

# ---- Headline numbers ----------------------------------------------------------
total_goals = df["home_goals"] + df["away_goals"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Matches", f"{len(df):,}")
c2.metric("Home / Draw / Away",
          " / ".join(f"{(df['result'] == o).mean():.0%}" for o in config.OUTCOMES))
c3.metric("Goals per match", f"{total_goals.mean():.2f}")
c4.metric("Over 2.5 goals", f"{(total_goals > 2.5).mean():.0%}")
c5.metric("Both teams scored",
          f"{((df['home_goals'] > 0) & (df['away_goals'] > 0)).mean():.0%}")

fav = df[p_cols].max(axis=1)
fav_outcome = df[p_cols].to_numpy().argmax(axis=1)
fav_won = pd.Series(
    [config.OUTCOMES[i] for i in fav_outcome], index=df.index
) == df["result"]
valid = fav.notna()
m1, m2, m3 = st.columns(3)
m1.metric("Mean favorite probability", f"{fav[valid].mean():.1%}")
m2.metric("Favorite win rate", f"{fav_won[valid].mean():.1%}")
if f"margin_{source}" in df.columns:
    m3.metric("Mean margin", f"{df[f'margin_{source}'].mean():.2%}")

# ---- Charts --------------------------------------------------------------------
tab_season, tab_goals, tab_teams, tab_upsets = st.tabs(
    ["By season", "Goals", "By team", "Upsets"]
)

with tab_season:
    per_season = df.groupby("season").agg(
        matches=("result", "size"),
        home=("result", lambda s: (s == "H").mean()),
        draw=("result", lambda s: (s == "D").mean()),
        away=("result", lambda s: (s == "A").mean()),
        goals=("home_goals", "mean"),
    )
    per_season["goals"] += df.groupby("season")["away_goals"].mean()
    fig = px.bar(
        per_season.reset_index().melt(
            id_vars=["season", "matches", "goals"],
            var_name="outcome", value_name="share",
        ),
        x="season", y="share", color="outcome",
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        per_season.style.format(
            {"home": "{:.1%}", "draw": "{:.1%}", "away": "{:.1%}", "goals": "{:.2f}"}
        ),
        use_container_width=True,
    )

with tab_goals:
    counts = (
        pd.crosstab(df["home_goals"].clip(upper=6), df["away_goals"].clip(upper=6),
                    normalize=True)
    )
    fig = px.imshow(
        counts,
        labels=dict(x="away goals", y="home goals", color="share"),
        text_auto=".1%",
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(height=460, title="Scoreline distribution (6 = 6+)")
    st.plotly_chart(fig, use_container_width=True)

with tab_teams:
    home = df.groupby("home_team").agg(
        home_matches=("result", "size"),
        home_wins=("result", lambda s: (s == "H").sum()),
        home_goals=("home_goals", "sum"),
    )
    away = df.groupby("away_team").agg(
        away_matches=("result", "size"),
        away_wins=("result", lambda s: (s == "A").sum()),
        away_goals=("away_goals", "sum"),
    )
    team = home.join(away, how="outer").fillna(0)
    team["matches"] = team["home_matches"] + team["away_matches"]
    team["wins"] = team["home_wins"] + team["away_wins"]
    team["win_rate"] = team["wins"] / team["matches"]
    team["goals_for"] = team["home_goals"] + team["away_goals"]
    show = team[["matches", "wins", "win_rate", "goals_for"]].sort_values(
        "win_rate", ascending=False
    )
    st.dataframe(
        show.astype({"matches": int, "wins": int, "goals_for": int})
        .style.format({"win_rate": "{:.1%}"}),
        use_container_width=True,
        height=480,
    )

with tab_upsets:
    st.markdown(
        "Matches where the realized outcome had the **lowest** market "
        "probability — the biggest surprises passing your filters."
    )
    with_p = df.dropna(subset=p_cols).copy()
    onehot = np.column_stack(
        [(with_p["result"] == o).to_numpy() for o in config.OUTCOMES]
    )
    with_p["realized_prob"] = with_p[p_cols].to_numpy()[onehot].reshape(-1)
    ups = with_p.nsmallest(20, "realized_prob")[
        ["date", "season", "home_team", "away_team",
         "home_goals", "away_goals", "result", "realized_prob"]
    ]
    ups["date"] = ups["date"].dt.date
    st.dataframe(
        ups.set_index("date").style.format({"realized_prob": "{:.1%}"}),
        use_container_width=True,
    )

st.download_button(
    "Download filtered matches (CSV)",
    df.to_csv(index=False).encode(),
    file_name="odds_lab_filtered.csv",
    mime="text/csv",
)
