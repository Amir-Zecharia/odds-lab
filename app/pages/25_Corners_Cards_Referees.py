import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import explain, load_matches, page_setup, season_filter
from src.match_stats import (
    CountModel,
    evaluate_lines,
    referee_multipliers,
    rolling_forecast,
    with_derived_stats,
)

page_setup("Corners, Cards & Referees", icon="🚩")
st.caption(
    "The match statistics the source has carried since 2000 and this "
    "platform never used. No liquid closing line disciplines these "
    "markets — so unlike match odds, here we are the only forecaster."
)

df = with_derived_stats(season_filter(load_matches()))

target = st.sidebar.selectbox(
    "Statistic",
    ["corners", "cards", "shots", "shots_target", "fouls"],
    format_func=lambda t: t.replace("_", " ").title(),
)
half_life = st.sidebar.select_slider(
    "Recency half-life (days)",
    options=[365, 730, 1460, 0],
    value=730,
    format_func=lambda d: "no decay" if d == 0 else f"{d // 365} year(s)",
    help="Corner and card rates drift across eras. Weighting recent matches "
    "more heavily measurably improves calibration.",
)

home_col, away_col = {
    "corners": ("home_corners", "away_corners"),
    "cards": ("home_cards", "away_cards"),
    "shots": ("home_shots", "away_shots"),
    "shots_target": ("home_shots_target", "away_shots_target"),
    "fouls": ("home_fouls", "away_fouls"),
}[target]
covered = df.dropna(subset=[home_col, away_col])
if covered.empty:
    st.warning("No data for this statistic in the current selection.")
    st.stop()

totals = covered[home_col] + covered[away_col]
c1, c2, c3 = st.columns(3)
c1.metric("Matches with data", f"{len(covered):,}")
c2.metric(f"Mean {target.replace('_', ' ')} per match", f"{totals.mean():.2f}")
c3.metric("Home share", f"{covered[home_col].sum() / totals.sum():.1%}",
          help="Home teams take more corners and collect fewer cards.")

# ---- the drift that motivates recency weighting ------------------------------
st.subheader("The rate drifts — which is why the model decays old matches")
by_season = covered.groupby("season").apply(
    lambda g: (g[home_col] + g[away_col]).mean()
)
fig = px.line(by_season.reset_index(name="per_match"), x="season", y="per_match",
              markers=True, labels={"per_match": f"{target} per match"})
fig.update_layout(height=360)
st.plotly_chart(fig, use_container_width=True)

# ---- forecast quality ----------------------------------------------------------
st.subheader("Can we forecast it?")
lines_by_target = {
    "corners": (8.5, 9.5, 10.5, 11.5),
    "cards": (2.5, 3.5, 4.5),
    "shots": (22.5, 25.5, 28.5),
    "shots_target": (7.5, 8.5, 9.5),
    "fouls": (19.5, 21.5, 23.5),
}


@st.cache_data(show_spinner="Rolling expanding-window forecast ...")
def _forecast(seasons: tuple, leagues: tuple, target: str, hl: int):
    data = with_derived_stats(load_matches())
    data = data[data["season"].isin(seasons) & data["league"].isin(leagues)]
    return rolling_forecast(
        data, target=target, min_train=400, refit_every=100,
        half_life_days=None if hl == 0 else hl,
    )


forecast = _forecast(
    tuple(sorted(df["season"].unique())), tuple(sorted(df["league"].unique())),
    target, half_life,
)
if forecast.empty:
    st.info("Not enough history in this selection to train a rolling forecast.")
else:
    scored = evaluate_lines(forecast, lines_by_target[target])
    flat = forecast.assign(
        pred_total=forecast["actual_total"].expanding().mean().shift(1).bfill()
    )
    baseline = evaluate_lines(flat, lines_by_target[target])
    table = scored.merge(baseline[["line", "log_loss"]], on="line",
                         suffixes=("", "_baseline"))
    table["beats_baseline"] = table["log_loss"] < table["log_loss_baseline"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=table["line"].astype(str), y=table["predicted_over_rate"],
                         name="predicted P(over)"))
    fig.add_trace(go.Bar(x=table["line"].astype(str), y=table["actual_over_rate"],
                         name="actual over rate"))
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=380, barmode="group",
                      xaxis_title=f"{target} line")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        table.set_index("line").style.format(
            {"predicted_over_rate": "{:.1%}", "actual_over_rate": "{:.1%}",
             "gap": "{:+.2%}", "log_loss": "{:.4f}", "log_loss_baseline": "{:.4f}"}
        ),
        use_container_width=True,
    )
    won = int(table["beats_baseline"].sum())
    st.caption(
        f"The team-strength model beats a flat league-average baseline on "
        f"**{won} of {len(table)} lines**, and by thin margins. Corners and "
        "cards are far closer to noise than intuition suggests — team "
        "tendencies are real but small next to match-to-match variance."
    )

# ---- referees ------------------------------------------------------------------
st.subheader("Do referees have persistent tendencies?")
with_ref = df.dropna(subset=["referee"])
if with_ref.empty:
    st.info("No referee data in this selection (the source records it from 2019-20).")
else:
    min_matches = st.slider("Minimum matches per referee", 10, 100, 30, 5)
    counts = with_ref["referee"].value_counts()
    busy = counts[counts >= min_matches].index
    mult = referee_multipliers(
        with_ref[with_ref["referee"].isin(busy)], target=target,
        by_league=True, by_season=True,
    )
    if not mult:
        st.info("Not enough referee coverage at this threshold.")
    else:
        series = pd.Series(mult).sort_values()
        ends = pd.concat([series.head(10), series.tail(10)])
        fig = px.bar(
            ends.reset_index(), x=0, y="index", orientation="h",
            labels={"0": f"{target} multiplier vs their own league-season", "index": ""},
        )
        fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
        fig.update_layout(height=560)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"{len(series)} referees with {min_matches}+ matches, spread "
            f"**{series.min():.2f}× to {series.max():.2f}×**. Each is measured "
            "against their own league and season, so this is the official's "
            "personal tendency — not the competition's or the era's."
        )

explain(
    "How to read this page",
    """
- **Why these markets are different.** Every other page here defers to the
  closing line, because the market prices match outcomes better than we
  can. No such line disciplines corners and cards — so these forecasts
  stand or fall on their own calibration, not on beating anyone.
- **The model.** Counts are near-Poisson, so expected corners = league
  baseline × how many this team generates × how many that team concedes,
  with every rate shrunk toward the average by sample size and old
  matches exponentially discounted.
- **Two confounds this page corrects**, both found the hard way: referee
  multipliers pooled across leagues really measure *Italy vs England*, and
  pooled across seasons they measure *2004 vs today* (cards have risen for
  two decades). Baselining within league **and** season leaves the person.
- **The honest verdict on forecasting them:** barely better than the
  league average. Team tendency in corners and cards is real but tiny
  against match-to-match noise — worth knowing before trusting any
  confident-sounding corner prediction, anywhere.
""",
)
