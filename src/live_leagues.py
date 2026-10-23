"""Live-only provider registry; historical research leagues stay unchanged."""
from src.config import LEAGUES
from src.live import SPORT_KEYS

ISRAEL_LEAGUE = "IL1"
LIVE_LEAGUES = {code: {"name": LEAGUES[code]["name"], "provider": "odds_api"}
                for code in SPORT_KEYS}
LIVE_LEAGUES[ISRAEL_LEAGUE] = {"name": "Ligat Ha'Al", "provider": "api_football"}


def live_league_name(code: str) -> str:
    return LIVE_LEAGUES[code]["name"]
