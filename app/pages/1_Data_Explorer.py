import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import load_matches, page_setup, season_filter
from src import descriptive

page_setup("Data Explorer", icon="🗺️")
st.caption(
    "The descriptive layer: what the raw data supports. The coverage matrix "
    "bounds what every other analysis can claim."
)

df = season_filter(load_matches())

tab_rates, tab_margin, tab_cov, tab_disp = st.tabs(
    ["Outcome base rates", "Bookmaker margins", "Coverage matrix", "Dispersion"]
)

with tab_rates:
    rates = descriptive.outcome_base_rates(df)
    st.markdown(
        f"Across the selection: **{rates['home_win'].mean():.1%} home / "
        f"{rates['draw'].mean():.1%} draw / {rates['away_win'].mean():.1%} away** "
        "— the naive baseline any prediction must beat."
    )
    fig = px.area(
        rates.reset_index(),
        x="season",
        y=["home_win", "draw", "away_win"],
        labels={"value": "share of matches", "variable": "outcome"},
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    draw = descriptive.draw_rate_by_season(df)
    fig = px.line(draw.reset_index(), x="season", y="draw_rate", markers=True)
    fig.update_yaxes(tickformat=".0%")
    st.subheader("Draw rate by season")
    st.plotly_chart(fig, use_container_width=True)

with tab_margin:
    st.markdown(
        "Mean overround (sum of implied probabilities − 1) per book per "
        "season. The two-decade decline reflects competition and lower "
        "margins industry-wide; Pinnacle is structurally the cheapest."
    )
    margins = descriptive.margin_by_season(df)
    fig = px.line(
        margins.reset_index().melt(id_vars="season", var_name="book", value_name="margin"),
        x="season",
        y="margin",
        color="book",
        markers=True,
    )
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)

with tab_cov:
    st.markdown(
        "Fraction of matches per season with complete H/D/A odds per book. "
        "**Non-negotiable context**: analyses involving a book are only valid "
        "where it has coverage (e.g. Pinnacle appears in 2012-13; closing "
        "lines for most books in 2019-20)."
    )
    cov = descriptive.coverage_matrix(df)
    fig = go.Figure(
        go.Heatmap(
            z=cov.T.values,
            x=cov.index,
            y=cov.columns.tolist(),
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar_title="coverage",
        )
    )
    fig.update_layout(height=520, xaxis_title="season", yaxis_title="book")
    st.plotly_chart(fig, use_container_width=True)

with tab_disp:
    st.markdown(
        "Cross-book dispersion of the home-win probability: how much do "
        "bookmakers disagree? (They rarely do — correlations across books "
        "exceed 0.98.)"
    )
    valid = df.dropna(subset=["disp_std"])
    fig = px.histogram(valid, x="disp_std", nbins=60, labels={"disp_std": "std of home-win prob across books"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        descriptive.dispersion_distribution(df).style.format("{:.4f}"),
        use_container_width=True,
    )

    st.subheader("Favorite win rate by odds band")
    st.markdown("The raw table behind the calibration analysis.")
    band = descriptive.favorite_win_rate_by_band(df)
    band.index = band.index.astype(str)
    st.dataframe(
        band.style.format(
            {"mean_implied": "{:.1%}", "win_rate": "{:.1%}", "gap": "{:+.1%}"}
        ),
        use_container_width=True,
    )
