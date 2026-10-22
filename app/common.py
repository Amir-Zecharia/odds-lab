"""Shared dashboard helpers: cached data loading and season filtering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.run_analyses import ANALYSIS_JSON, ENRICHED_PARQUET  # noqa: E402

# Bookmaker prefixes (as they appear in probability columns, p_<book>_H)
# mapped to display names. Shared by the leaderboard and bookmaker pages.
BOOKS = {
    "PS": "Pinnacle",
    "PSC": "Pinnacle closing",
    "B365": "Bet365",
    "B365C": "Bet365 closing",
    "WH": "William Hill",
    "WHC": "William Hill closing",
    "BW": "Bwin",
    "IW": "Interwetten",
    "VC": "VC Bet",
    "Avg": "Market average",
    "AvgC": "Closing average",
    "Max": "Best price",
    "MaxC": "Closing best price",
    "cons": "Consensus (weighted)",
}

# Caches are keyed on the file's mtime so a re-run of ingest/run_analyses
# is picked up on the next interaction without restarting the server.


@st.cache_data(show_spinner="Loading match data ...")
def _matches_file(mtime: float) -> pd.DataFrame:
    return pd.read_parquet(ENRICHED_PARQUET)


def load_matches() -> pd.DataFrame:
    if not ENRICHED_PARQUET.exists():
        st.error(
            "No processed data found. Run `python -m src.ingest` and then "
            "`python -m src.run_analyses` first."
        )
        st.stop()
    return _matches_file(ENRICHED_PARQUET.stat().st_mtime)


@st.cache_data
def _analysis_file(mtime: float) -> dict:
    return json.loads(ANALYSIS_JSON.read_text())


def load_analysis(league: str | None = None) -> dict:
    """Per-league precomputed analyses (defaults to the sidebar selection)."""
    if not ANALYSIS_JSON.exists():
        st.error("Precomputed analyses missing. Run `python -m src.run_analyses`.")
        st.stop()
    data = _analysis_file(ANALYSIS_JSON.stat().st_mtime)
    if "leagues" not in data:  # pre-multi-league file
        return data
    league = league or st.session_state.get("league", config.DEFAULT_LEAGUE)
    per_league = data["leagues"].get(league)
    if per_league is None:
        st.error(f"No precomputed analyses for {league}. Re-run src.run_analyses.")
        st.stop()
    return {**per_league, "computed_at": data.get("computed_at", "")}


def league_selector() -> str:
    """Sidebar league radio (incl. pooled "All leagues"); the choice also
    drives load_analysis()."""
    return st.sidebar.radio(
        "League",
        config.league_options(),
        format_func=config.league_display_name,
        key="league",
    )


def league_filter(df: pd.DataFrame) -> pd.DataFrame:
    if "league" not in df.columns:
        return df
    choice = league_selector()
    if choice == config.ALL_LEAGUES:
        return df
    return df[df["league"] == choice]


def season_filter(df: pd.DataFrame, key: str = "seasons") -> pd.DataFrame:
    """Sidebar league + season-range selectors (league first)."""
    df = league_filter(df)
    seasons = sorted(df["season"].unique())
    preset = st.sidebar.radio(
        "Season range",
        ["All seasons", "Last 3 seasons", "Custom"],
        key=f"{key}_preset",
    )
    if preset == "All seasons":
        selected = seasons
    elif preset == "Last 3 seasons":
        selected = seasons[-3:]
    else:
        lo, hi = st.sidebar.select_slider(
            "Seasons",
            options=seasons,
            value=(seasons[0], seasons[-1]),
            key=f"{key}_slider",
        )
        selected = [s for s in seasons if lo <= s <= hi]
    st.sidebar.caption(
        f"{len(selected)} season(s), "
        f"{df['season'].isin(selected).sum():,} matches"
    )
    return df[df["season"].isin(selected)]


def page_setup(title: str, icon: str = "📊") -> None:
    st.set_page_config(page_title=f"{title} · Odds Lab", page_icon=icon, layout="wide")
    from src.bundle import bootstrap

    try:
        bootstrap()
    except ValueError as exc:
        st.error(f"Offline data bundle could not be verified: {exc}")
        st.stop()
    st.title(title)


def explain(title: str, body: str) -> None:
    """Consistent 'how to read this' expander used at the bottom of pages."""
    with st.expander(f"📖 {title}"):
        st.markdown(body)
