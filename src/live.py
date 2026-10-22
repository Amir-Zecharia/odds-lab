"""Live odds for upcoming fixtures via The Odds API (free tier).

Applies the platform's validated recipe to matches that haven't been
played yet: Shin de-margin each book, then prefer the sharpest quote
(Pinnacle) and fall back to the cross-book average — the rule that won
the Consensus Builder race.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from src.probabilities import demargin_shin

API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
SPORT_KEY = "soccer_epl"
SHARP_BOOK_KEY = "pinnacle"
# The Odds API v4 sport keys (https://the-odds-api.com/sports-odds-data/sports-apis.html).
SPORT_KEYS = {
    "E0": "soccer_epl", "E1": "soccer_efl_champ",
    "SP1": "soccer_spain_la_liga", "D1": "soccer_germany_bundesliga",
    "I1": "soccer_italy_serie_a", "F1": "soccer_france_ligue_one",
}


def fetch_events(api_key: str, sport: str = SPORT_KEY) -> tuple[list, dict]:
    """Return (events, quota headers). Raises for HTTP errors."""
    resp = requests.get(
        API_URL.format(sport=sport),
        params={
            "apiKey": api_key,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        },
        timeout=30,
    )
    resp.raise_for_status()
    quota = {
        "used": resp.headers.get("x-requests-used"),
        "remaining": resp.headers.get("x-requests-remaining"),
    }
    payload = resp.json()
    if not isinstance(payload, list):
        raise ValueError("The Odds API returned an invalid events payload")
    return payload, quota


def _h2h_odds(bookmaker: dict, home: str, away: str) -> list[float] | None:
    """[H, D, A] decimal odds from one bookmaker's h2h market."""
    if not isinstance(bookmaker, dict) or not isinstance(bookmaker.get("markets"), list):
        return None
    for market in bookmaker["markets"]:
        if not isinstance(market, dict) or market.get("key") != "h2h":
            continue
        outcomes = market.get("outcomes")
        if not isinstance(outcomes, list):
            continue
        prices = {o.get("name"): o.get("price") for o in outcomes
                  if isinstance(o, dict) and isinstance(o.get("name"), str)}
        values = [prices.get(home), prices.get("Draw"), prices.get(away)]
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float))
               for v in values):
            continue
        odds = np.asarray(values, dtype=float)
        if np.isfinite(odds).all() and (odds >= 1.01).all():
            return odds.tolist()
    return None


def parse_events(events: list) -> pd.DataFrame:
    """One row per fixture: kickoff, teams, prediction, source, book count."""
    rows = []
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict):
            continue
        home, away = event.get("home_team"), event.get("away_team")
        if not all(isinstance(t, str) and t.strip() for t in (home, away)) or home == away:
            continue
        timestamp = event.get("commence_time")
        if not isinstance(timestamp, str):
            continue
        try:
            kickoff = pd.Timestamp(timestamp)
            if pd.isna(kickoff) or kickoff.tzinfo is None:
                continue
            kickoff = kickoff.tz_convert("UTC")
        except (ValueError, TypeError, OverflowError):
            continue
        bookmakers = event.get("bookmakers")
        if not isinstance(bookmakers, list):
            continue
        book_probs, sharp = {}, None
        for bookmaker in bookmakers:
            odds = _h2h_odds(bookmaker, home, away)
            if odds is None or not isinstance(bookmaker.get("key"), str):
                continue
            probs = demargin_shin(np.array(odds, float))
            if not np.isfinite(probs).all() or (probs < 0).any() or probs.sum() <= 0:
                continue
            probs = probs / probs.sum()
            book_probs[bookmaker["key"]] = probs
            if bookmaker["key"] == SHARP_BOOK_KEY:
                sharp = probs
        if not book_probs:
            continue
        if sharp is not None:
            prediction, source = sharp, "Pinnacle"
        else:
            prediction = np.mean(list(book_probs.values()), axis=0)
            prediction = prediction / prediction.sum()
            source = f"average of {len(book_probs)} books"
        rows.append(
            {
                "event_id": event.get("id"),
                "kickoff": kickoff,
                "home_team": home,
                "away_team": away,
                "p_home": float(prediction[0]),
                "p_draw": float(prediction[1]),
                "p_away": float(prediction[2]),
                "source": source,
                "books": len(book_probs),
            }
        )
    return pd.DataFrame(rows).sort_values("kickoff") if rows else pd.DataFrame(
        columns=[
            "event_id", "kickoff", "home_team", "away_team",
            "p_home", "p_draw", "p_away", "source", "books",
        ]
    )


def fetch_scores(api_key: str, sport: str = SPORT_KEY, days_from: int = 3) -> tuple[list, dict]:
    """Fetch completed and recent scores within the provider's three-day window."""
    if isinstance(days_from, bool) or not isinstance(days_from, int) or not 1 <= days_from <= 3:
        raise ValueError("days_from must be an integer between 1 and 3")
    response = requests.get(
        "https://api.the-odds-api.com/v4/sports/{sport}/scores".format(sport=sport),
        params={"apiKey": api_key, "daysFrom": days_from}, timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("The Odds API returned an invalid scores payload")
    return payload, {"used": response.headers.get("x-requests-used"),
                     "remaining": response.headers.get("x-requests-remaining"),
                     "last": response.headers.get("x-requests-last")}


def parse_completed_scores(events: list) -> pd.DataFrame:
    """Validate finals independently; one malformed neighbor cannot discard valid results."""
    columns = ["event_id", "kickoff", "home_team", "away_team", "home_goals",
               "away_goals", "result", "provider_updated_at"]
    rows = []
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict) or event.get("completed") is not True:
            continue
        identity = event.get("id")
        home, away = event.get("home_team"), event.get("away_team")
        if not all(isinstance(v, str) and v.strip() for v in (identity, home, away)) or home == away:
            continue
        try:
            kickoff = pd.Timestamp(event.get("commence_time"))
            if pd.isna(kickoff) or kickoff.tzinfo is None:
                continue
            kickoff = kickoff.tz_convert("UTC")
            scores = event.get("scores")
            if not isinstance(scores, list) or len(scores) != 2:
                continue
            named = {}
            for score in scores:
                if not isinstance(score, dict) or score.get("name") in named:
                    raise ValueError("invalid score")
                value = score.get("score")
                if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).isascii() or not str(value).isdigit():
                    raise ValueError("invalid score")
                named[score.get("name")] = int(value)
            hg, ag = named[home], named[away]
            updated = pd.NaT
            if event.get("last_update") is not None:
                updated = pd.Timestamp(event["last_update"])
                if pd.isna(updated) or updated.tzinfo is None:
                    continue
                updated = updated.tz_convert("UTC")
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
        rows.append(dict(event_id=identity, kickoff=kickoff, home_team=home, away_team=away,
                         home_goals=hg, away_goals=ag, result="H" if hg > ag else "A" if ag > hg else "D",
                         provider_updated_at=updated))
    return pd.DataFrame(rows, columns=columns)
