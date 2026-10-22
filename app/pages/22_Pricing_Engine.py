import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import explain, load_matches, page_setup, season_filter
from src import config
from src.models import DixonColes
from src.pricing_engine import RESULTS_JSON, price_match

page_setup("Pricing Engine", icon="🏭")
st.caption(
    "A bookmaker-style origination desk: one scoreline distribution prices "
    "every market at once — 1X2, Asian handicap, over/under, correct score — "
    "so they can never contradict each other. Pick a fixture and watch it "
    "price; then see how far the engine trails the real desks per market."
)

df = season_filter(load_matches())

# ---- interactive origination desk -------------------------------------------
st.subheader("Price a fixture")
seasons = sorted(df["season"].unique())
season = st.selectbox("Season to price", list(reversed(seasons)))
season_df = df[df["season"] == season]
fixtures = season_df.sort_values("date", ascending=False)
labels = {
    int(r.match_id): f"{r.date.date()} · {r.home_team} vs {r.away_team}"
    for r in fixtures.itertuples()
}
match_id = st.selectbox(
    "Fixture", list(labels), format_func=labels.get,
    help="The engine trains on this fixture's league, strictly on matches "
    "before its season — then prices it blind.",
)
row = df[df["match_id"] == match_id].iloc[0]


@st.cache_data(show_spinner="Fitting the goal model (trailing 6 seasons) ...")
def fit_for(league: str, season: str):
    data = load_matches()
    pool = data[(data["league"] == league) & (data["season"] < season)]
    window = sorted(pool["season"].unique())[-6:]
    return DixonColes().fit(pool[pool["season"].isin(window)])


model = fit_for(row["league"], season)
ah_line = row.get("ah_line", float("nan"))
priced = price_match(model, row["home_team"], row["away_team"],
                     ah_line if np.isfinite(ah_line) else None)

c1, c2, c3, c4 = st.columns(4)
c1.metric("P(home)", f"{priced['p_home']:.1%}",
          help=f"Market consensus: {row.get('p_cons_H', float('nan')):.1%}")
c2.metric("P(draw)", f"{priced['p_draw']:.1%}",
          help=f"Market consensus: {row.get('p_cons_D', float('nan')):.1%}")
c3.metric("P(over 2.5)", f"{priced['p_over25']:.1%}",
          help=f"Market: {row.get('p_over_Avg', float('nan')):.1%}")
if "p_ah_cover" in priced:
    c4.metric(f"P(AH {ah_line:+.2f} covers)", f"{priced['p_ah_cover']:.1%}",
              help=f"Market: {row.get('p_ahcover_Avg', float('nan')):.1%}")
else:
    c4.metric("BTTS", f"{priced['p_btts']:.1%}")

grid = model.score_grid(row["home_team"], row["away_team"])[:6, :6]
fig = px.imshow(
    grid,
    labels=dict(x=f"{row['away_team']} goals", y=f"{row['home_team']} goals",
                color="probability"),
    text_auto=".1%",
    color_continuous_scale="Blues",
)
fig.update_layout(height=430, title="Simulated scoreline distribution")
st.plotly_chart(fig, use_container_width=True)
actual = f"{int(row['home_goals'])}-{int(row['away_goals'])}"
st.caption(
    f"Every price above is a sum over this one grid. Actual result: "
    f"**{actual}** ({row['result']})."
)

# ---- the race ------------------------------------------------------------------
st.subheader("Engine vs the real desks, market by market")
if RESULTS_JSON.exists():
    results = json.loads(RESULTS_JSON.read_text())
    names = {"1x2": "1X2 (match odds)", "totals": "Over/under 2.5", "ah": "Asian handicap"}
    rows = [
        {"market": names.get(k, k), **v} for k, v in results["markets"].items()
    ]
    table = pd.DataFrame(rows)
    fig = go.Figure(
        go.Bar(
            x=table["market"],
            y=table["gap"],
            error_y=dict(type="data", symmetric=False,
                         array=table["hi"] - table["gap"],
                         arrayminus=table["gap"] - table["lo"]),
            customdata=table["n"],
            hovertemplate="%{x}: +%{y:.4f} nats (n=%{customdata:,})",
        )
    )
    fig.update_layout(
        yaxis_title="log-loss gap: engine − market (lower = closer)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        table.set_index("market").style.format(
            {"model_log_loss": "{:.4f}", "market_log_loss": "{:.4f}",
             "gap": "{:+.4f}", "lo": "{:+.4f}", "hi": "{:+.4f}",
             "gap_pct_of_market": "{:+.2%}"}
        ),
        use_container_width=True,
    )
else:
    st.caption("Run `python -m src.pricing_engine` to compute the race.")

explain(
    "How to read this page",
    """
- **The grid is the desk.** Real origination works exactly like this: a
  goal model produces the joint scoreline distribution, and every market
  is a sum over its cells. That's why a book's 1X2, handicap and totals
  prices never contradict each other — they share one underlying model.
- **Quarter-line handicaps** (−0.25, −0.75, …) split your stake across the
  two neighbouring lines; the engine prices them with the exact zero-EV
  identity including pushes and half-stakes.
- **The race** refits the goal model before each test season (trailing six
  seasons, strictly past data) and compares its prices to the de-margined
  book prices on identical matches. The gap is the *price of ignoring the
  market* — and comparing gaps across markets shows where public
  information goes furthest: goals markets depend more on stable team
  strength, 1X2 on transient news the books see and we don't.
- The engine is honest but simple (Dixon-Coles). Real desks layer xG,
  lineups and market flow on top — which is exactly why the gap exists.
""",
)
