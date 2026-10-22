"""Odds-free match features, computed strictly chronologically.

Every feature for match t uses only matches that finished before t —
enforced by iterating the frame in date order and reading team state
before updating it with the match result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 60.0  # Elo points added to home team's rating in expectation
FORM_WINDOW = 5


def add_rounds(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the matchweek ("round", 1-38) of every match.

    The source data carries no round column, so it is reconstructed as each
    team's match number within the season. Postponements can push the two
    teams' counters slightly apart; the round is their rounded mean.
    """
    df = df.sort_values("date").copy()
    counters: dict[tuple[str, str], int] = {}
    rounds = []
    for row in df.itertuples():
        key_h = (row.season, row.home_team)
        key_a = (row.season, row.away_team)
        h = counters.get(key_h, 0) + 1
        a = counters.get(key_a, 0) + 1
        counters[key_h] = h
        counters[key_a] = a
        rounds.append(int((h + a) / 2 + 0.5))
    df["round"] = rounds
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach elo_home/elo_away/elo_diff, form_diff, rest_diff.

    - Elo: standard update with fixed home advantage, K=20, goal-difference
      multiplier per Hvattum & Arntzen (2010).
    - Form: mean points over each team's last 5 matches (NaN until played 5).
    - Rest: days since each team's previous match (NaN for first match).
    """
    df = df.sort_values("date").copy()
    ratings: dict[str, float] = {}
    recent_points: dict[str, list[float]] = {}
    last_played: dict[str, pd.Timestamp] = {}

    elo_h, elo_a, form_h, form_a, rest_h, rest_a = ([] for _ in range(6))

    for row in df.itertuples():
        home, away = row.home_team, row.away_team
        rh = ratings.get(home, ELO_START)
        ra = ratings.get(away, ELO_START)
        elo_h.append(rh)
        elo_a.append(ra)

        for team, out in ((home, form_h), (away, form_a)):
            pts = recent_points.get(team, [])
            out.append(np.mean(pts[-FORM_WINDOW:]) if len(pts) >= FORM_WINDOW else np.nan)

        for team, out in ((home, rest_h), (away, rest_a)):
            prev = last_played.get(team)
            out.append((row.date - prev).days if prev is not None else np.nan)

        # --- update state with this match's result (after recording) ---
        expected_home = 1.0 / (1.0 + 10 ** (-(rh + ELO_HOME_ADV - ra) / 400.0))
        score_home = {"H": 1.0, "D": 0.5, "A": 0.0}[row.result]
        goal_diff = abs(row.home_goals - row.away_goals)
        k = ELO_K * np.sqrt(max(goal_diff, 1))
        delta = k * (score_home - expected_home)
        ratings[home] = rh + delta
        ratings[away] = ra - delta

        points = {"H": (3.0, 0.0), "D": (1.0, 1.0), "A": (0.0, 3.0)}[row.result]
        recent_points.setdefault(home, []).append(points[0])
        recent_points.setdefault(away, []).append(points[1])
        last_played[home] = row.date
        last_played[away] = row.date

    df["elo_home"] = elo_h
    df["elo_away"] = elo_a
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["form_home"] = form_h
    df["form_away"] = form_a
    df["form_diff"] = df["form_home"] - df["form_away"]
    df["rest_home"] = rest_h
    df["rest_away"] = rest_a
    # Cap rest at 30 days: summer/winter breaks are schedule, not fatigue.
    df["rest_diff"] = (
        df["rest_home"].clip(upper=30) - df["rest_away"].clip(upper=30)
    )
    return df
