"""The rating zoo: three published team-strength systems beyond Elo.

Each system answers "how strong is this team?" from a different part of
the data, which is exactly what an ensemble wants — informative
disagreement:

- **Pi-ratings** (Constantinou & Fenton 2013): sequential like Elo but
  football-native — separate home/away ratings per team, updated by the
  *goal-difference* prediction error with diminishing weight on blowouts.
- **Massey** (1997): one least-squares solve over a window of score
  margins. No recency preference — a photograph, not a diary.
- **Colley**: Massey's structure but deliberately blind to scores — wins
  and losses only (draws count half each), Laplace-smoothed.

All features are strictly pre-match: sequential systems update after
recording, and the solved systems refit per matchday from matches strictly
before it. Ratings are league-internal, mirroring src/features.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Pi-rating constants from Constantinou & Fenton (2013).
PI_LAMBDA = 0.035   # learning rate
PI_GAMMA = 0.7      # cross-update of the "other venue" rating
PI_C = 3.0          # error dampening: psi = c * log10(1 + |error|)

# Solved systems refit from a trailing window; early matches lack ratings.
WINDOW_DAYS = 730
MIN_POOL_MATCHES = 150


def _pi_update(rating_error: float) -> float:
    """Diminishing-returns error weight: blowouts move ratings sublinearly."""
    return PI_C * np.log10(1.0 + abs(rating_error)) * np.sign(rating_error)


def add_pi_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-match Pi-rating expected goal difference (``pi_diff``).

    Each team carries a home rating and an away rating in goal-difference
    units; the match prediction is home team's home rating minus away
    team's away rating. Updates happen after the prediction is recorded.
    """
    df = df.sort_values("date").copy()
    home_r: dict[str, float] = {}
    away_r: dict[str, float] = {}
    pred = []
    for row in df.itertuples():
        rh = home_r.get(row.home_team, 0.0)
        ra = away_r.get(row.away_team, 0.0)
        expected_gd = rh - ra
        pred.append(expected_gd)

        error = (row.home_goals - row.away_goals) - expected_gd
        step = PI_LAMBDA * _pi_update(error)
        home_r[row.home_team] = rh + step
        away_r[row.home_team] = away_r.get(row.home_team, 0.0) + PI_GAMMA * step
        away_r[row.away_team] = ra - step
        home_r[row.away_team] = home_r.get(row.away_team, 0.0) - PI_GAMMA * step

    df["pi_diff"] = pred
    return df


def massey_solve(pool: pd.DataFrame) -> dict[str, float]:
    """Least-squares Massey ratings from a pool of matches (margin-based).

    Solves M r = p with the last row replaced by the sum-to-zero
    constraint, the standard fix for M's rank deficiency.
    """
    teams = sorted(set(pool["home_team"]) | set(pool["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    M = np.zeros((n, n))
    p = np.zeros(n)
    for row in pool.itertuples():
        i, j = idx[row.home_team], idx[row.away_team]
        M[i, i] += 1
        M[j, j] += 1
        M[i, j] -= 1
        M[j, i] -= 1
        margin = row.home_goals - row.away_goals
        p[i] += margin
        p[j] -= margin
    M[-1, :] = 1.0
    p[-1] = 0.0
    ratings = np.linalg.lstsq(M, p, rcond=None)[0]
    return dict(zip(teams, ratings))


def colley_solve(pool: pd.DataFrame) -> dict[str, float]:
    """Colley ratings: wins/losses only (draws = half each), smoothed."""
    teams = sorted(set(pool["home_team"]) | set(pool["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    C = 2.0 * np.eye(n)
    b = np.ones(n)
    for row in pool.itertuples():
        i, j = idx[row.home_team], idx[row.away_team]
        C[i, i] += 1
        C[j, j] += 1
        C[i, j] -= 1
        C[j, i] -= 1
        if row.result == "H":
            wi, wj = 1.0, 0.0
        elif row.result == "A":
            wi, wj = 0.0, 1.0
        else:
            wi = wj = 0.5
        b[idx[row.home_team]] += (wi - wj) / 2.0
        b[idx[row.away_team]] += (wj - wi) / 2.0
    ratings = np.linalg.solve(C, b)
    return dict(zip(teams, ratings))


def add_solved_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-match Massey and Colley rating differences.

    Refit per matchday from the trailing two-year window of matches that
    finished strictly earlier — cheap (a 20-40 team linear solve) and
    leak-free by construction. NaN until the pool is meaningful.
    """
    df = df.sort_values("date").copy()
    massey_diff = np.full(len(df), np.nan)
    colley_diff = np.full(len(df), np.nan)
    dates = df["date"].to_numpy()
    positions = np.arange(len(df))

    for day, day_pos in pd.Series(positions, index=dates).groupby(level=0):
        pool = df.iloc[: day_pos.iloc[0]]
        pool = pool[pool["date"] >= day - pd.Timedelta(days=WINDOW_DAYS)]
        if len(pool) < MIN_POOL_MATCHES:
            continue
        massey = massey_solve(pool)
        colley = colley_solve(pool)
        for pos in day_pos:
            home, away = df.iloc[pos]["home_team"], df.iloc[pos]["away_team"]
            if home in massey and away in massey:
                massey_diff[pos] = massey[home] - massey[away]
                colley_diff[pos] = colley[home] - colley[away]

    df["massey_diff"] = massey_diff
    df["colley_diff"] = colley_diff
    return df


def add_zoo_features(df: pd.DataFrame) -> pd.DataFrame:
    """All zoo ratings, computed league-internally like src/features.py."""
    per_league = []
    group_key = "league" if "league" in df.columns else None
    groups = df.groupby(group_key) if group_key else [(None, df)]
    for _, group in groups:
        group = add_pi_ratings(group)
        group = add_solved_ratings(group)
        per_league.append(group)
    out = pd.concat(per_league, ignore_index=True)
    return out.sort_values(
        [c for c in ("league", "date", "home_team") if c in out.columns]
    ).reset_index(drop=True)
