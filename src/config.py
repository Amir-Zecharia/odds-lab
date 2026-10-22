"""Central configuration: paths, seasons, bookmaker metadata."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = PROJECT_ROOT / "data" / "bundled"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MATCHES_PARQUET = PROCESSED_DIR / "matches.parquet"

# League registry: adding a league = one entry here (football-data.co.uk
# division code -> display name + season-size bounds). League sizes change
# over the years (Serie A grew 306->380 in 2004-05, Ligue 1 shrank back to
# 306 in 2023-24, COVID truncated Ligue 1 2019-20 at 279), so validation
# uses ``max_matches`` as the historical upper bound for every season and
# ``recent_matches`` as the exact requirement for REQUIRED_RECENT_SEASONS.
LEAGUES = {
    "E0": {"name": "Premier League", "max_matches": 380, "recent_matches": 380},
    "E1": {"name": "Championship", "max_matches": 552, "recent_matches": 552},
    "SP1": {"name": "La Liga", "max_matches": 380, "recent_matches": 380},
    "D1": {"name": "Bundesliga", "max_matches": 306, "recent_matches": 306},
    "I1": {"name": "Serie A", "max_matches": 380, "recent_matches": 380},
    "F1": {"name": "Ligue 1", "max_matches": 380, "recent_matches": 306},
}
DEFAULT_LEAGUE = "E0"
LEAGUE_CODE = DEFAULT_LEAGUE  # back-compat alias
LEAGUE_NAME = LEAGUES[DEFAULT_LEAGUE]["name"]

# Pooled pseudo-league: every analysis is also available across all
# leagues combined. Any league added to LEAGUES joins the pool automatically.
ALL_LEAGUES = "ALL"


def league_options() -> list[str]:
    """Selector codes: each configured league plus the pooled option."""
    return list(LEAGUES) + [ALL_LEAGUES]


def league_display_name(code: str) -> str:
    return "All leagues" if code == ALL_LEAGUES else LEAGUES[code]["name"]

FIRST_SEASON_START = 2000
LAST_SEASON_START = 2025  # 2025-26 season

# Seasons whose data must be present and complete (hard requirement,
# enforced per league).
REQUIRED_RECENT_SEASONS = ["2023-24", "2024-25", "2025-26"]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"

# Individual bookmaker odds column prefixes that appear across
# football-data.co.uk seasons (each with H/D/A suffixes).
BOOK_PREFIXES = [
    "B365", "BF", "BS", "BW", "GB", "IW", "LB", "PS", "SB",
    "SJ", "SO", "SY", "VC", "WH", "1XB", "BFE",
]

# Closing-odds variants (from 2019-20 the source records both the
# midweek price, e.g. PSH, and the closing price, e.g. PSCH).
CLOSING_PREFIXES = [p + "C" for p in BOOK_PREFIXES] + ["AvgC", "MaxC"]

# Books singled out for the sharp-vs-soft analyses.
SHARP_BOOK = "PS"      # Pinnacle
SOFT_BOOK = "B365"     # Bet365

OUTCOMES = ["H", "D", "A"]


def season_label(start_year: int) -> str:
    """2004 -> '2004-05'"""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def season_url_code(start_year: int) -> str:
    """2004 -> '0405'"""
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def all_seasons() -> list[int]:
    return list(range(FIRST_SEASON_START, LAST_SEASON_START + 1))
